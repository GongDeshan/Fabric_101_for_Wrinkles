"""One optimization epoch per Polyscope callback, with visible progress."""
import math
import time
import numpy as np
from polyscope import imgui


class StretchTrainingProgress:
    def __init__(self, env):
        self.env = env
        self.running = False
        self.completed = 0
        self.target_epochs = int(env["NUM_EPOCH"])
        self.loss_history = []
        self.epoch_seconds = []
        self.force_history = []
        self.follow_latest_force = True
        self.force_epoch = 1
        self.log_loss_axis = False
        self.loss = None
        self.best_loss = None
        self.best_stiffness = None
        self.message = 'Ready'
        self.defer_step = False
        from stretch_record_writer import StretchRecordWriter
        self.record_writer = StretchRecordWriter(env)
        self.save_records = True
        self.save_error = ''


    def start(self):
        if self.completed >= self.target_epochs:
            self.target_epochs = self.completed + int(self.env['NUM_EPOCH'])
        self.env['run'] = False
        self.running = True
        self.defer_step = True
        self.message = 'Optimizing'

    def step(self):
        if not self.running:
            return
        if self.defer_step:
            self.defer_step = False
            return
        e = self.env
        try:
            epoch_started = time.perf_counter()
            evaluated_stiffness = float(e['parm'].forward().detach().item())
            loss = e['optimizer'].step(e['loss_fn'])
            epoch_elapsed = time.perf_counter() - epoch_started
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite loss')
            pred = np.asarray(e['_stretch_last_prediction'], dtype=np.float64).copy()
            gt = e['forces_gt'].detach().cpu().numpy()
            if pred.shape != gt.shape or not np.isfinite(pred).all():
                raise RuntimeError('Invalid prediction curve')
            mse = float(np.mean((pred-gt)**2))
            if not math.isclose(mse, value, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError('Prediction curve does not match this epoch loss')
            self.force_history.append(dict(pred=pred, stiffness=evaluated_stiffness, loss=value))
            self.loss = value
            self.loss_history.append(value)
            self.epoch_seconds.append(epoch_elapsed)
            self.completed += 1
            if self.best_loss is None or value < self.best_loss:
                self.best_loss = value
                self.best_stiffness = evaluated_stiffness
            if self.save_records:
                try:
                    self.record_writer.save(self.completed, pred, evaluated_stiffness, value, epoch_elapsed,
                        float(e['parm'].forward().detach().item()))
                    self.save_error = ''
                except Exception as exc:
                    self.save_error = 'Saving failed: ' + str(exc)
                    self.save_records = False
                    import traceback
                    traceback.print_exc()

            if self.completed >= self.target_epochs:
                self.running = False
                self.message = 'Completed'
        except Exception as exc:
            self.running = False
            self.message = 'Optimization failed: ' + str(exc)
            import traceback
            traceback.print_exc()

    def plot_values(self):
        values = np.asarray(self.loss_history, dtype=np.float64)
        if self.log_loss_axis:
            values = np.log10(np.maximum(values, np.finfo(np.float64).tiny))
        limit = np.finfo(np.float32).max
        return np.asarray(np.clip(values, -limit, limit), dtype=np.float32)

    def draw(self):
        e = self.env
        imgui.Text('Optimization progress')
        _, self.save_records = imgui.Checkbox('Save training records', self.save_records)
        from stretch_logging_theme import set_logging_theme
        set_logging_theme(self.save_records)
        if not self.save_records:
            imgui.Text('LOG SAVING IS OFF - training records are not being written.')
        if self.record_writer.directory is not None:
            imgui.Text('Records: ' + str(self.record_writer.directory))
        else:
            imgui.Text('Records directory is created after the first saved epoch.')
        imgui.Text('Disabled epochs are not written; existing files are kept.')
        if self.save_error:
            imgui.Text(self.save_error)

        imgui.Text(f'Epoch: {self.completed} / {self.target_epochs}')
        if self.epoch_seconds:
            average = sum(self.epoch_seconds) / len(self.epoch_seconds)
            remaining = max(0, self.target_epochs - self.completed) * average
            imgui.Text(f'Last epoch: {self.epoch_seconds[-1]:.2f} s/epoch')
            imgui.Text(f'Mean epoch: {average:.2f} s/epoch')
            label = 'Remaining compute (paused)' if not self.running and self.completed < self.target_epochs else 'Estimated remaining compute'
            imgui.Text(f'{label}: {remaining/60:.1f} min')
        else:
            imgui.Text('Epoch speed: available after the first epoch')
        imgui.Text('Latest loss (MSE): ' + ('--' if self.loss is None else f'{self.loss:.10g}'))
        imgui.Text('Best loss (MSE): ' + ('--' if self.best_loss is None else f'{self.best_loss:.10g}'))
        imgui.Text(f'Current axial stiffness: {float(e["parm"].forward().detach().item()):.10g} Pa')
        if self.best_stiffness is not None:
            imgui.Text(f'Stiffness at best loss: {self.best_stiffness:.10g} Pa')
        imgui.Text('Loss vs epoch')
        _, self.log_loss_axis = imgui.Checkbox('Log10 loss axis', self.log_loss_axis)
        if self.loss_history:
            imgui.PlotLines('##stretch_loss_epoch', self.plot_values(),
                overlay_text='log10(MSE)' if self.log_loss_axis else 'MSE',
                graph_size=(300.0, 160.0))
            imgui.Text(f'Epoch 1 (left) to {len(self.loss_history)} (right)')
        else:
            imgui.Text('The curve appears after the first completed epoch.')
        imgui.Text('Predicted force vs GT')
        if self.force_history:
            _, self.follow_latest_force = imgui.Checkbox('Show latest epoch force', self.follow_latest_force)
            if self.follow_latest_force:
                self.force_epoch = len(self.force_history)
            else:
                _, self.force_epoch = imgui.InputInt('Force curve epoch', self.force_epoch)
                self.force_epoch = max(1, min(len(self.force_history), self.force_epoch))
            record = self.force_history[self.force_epoch-1]
            imgui.Text(f'Curve epoch {self.force_epoch}; MSE={record["loss"]:.10g}; evaluated stiffness={record["stiffness"]:.10g} Pa')
            from stretch_force_plot import draw_force_overlay
            draw_force_overlay(e['real_load_pos_forces'][0], e['forces_gt'].detach().cpu().numpy(), record['pred'])
        else:
            imgui.Text('Prediction appears after the first completed epoch.')
        imgui.Text(self.message)
        if self.running and imgui.Button('Pause optimization'):
            self.running = False
            self.message = 'Paused'
        imgui.Text('The panel refreshes between epochs. Each epoch evaluates loss before its update.')
