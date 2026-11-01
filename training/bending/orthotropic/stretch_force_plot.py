"""Overlay the evaluated epoch's force curve and GT in the existing panel."""
import numpy as np
from polyscope import imgui


def curve_limits(x, gt, pred):
    x = np.asarray(x, dtype=float)
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if x.ndim != 1 or not (x.shape == gt.shape == pred.shape):
        raise ValueError('Displacement, GT and prediction must have matching vector shapes.')
    if not all(np.isfinite(v).all() for v in (x, gt, pred)):
        raise ValueError('Nonfinite force curve.')
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    ymin, ymax = min(0., float(gt.min()), float(pred.min())), max(0., float(gt.max()), float(pred.max()))
    pad = max((ymax-ymin)*.08, 1e-8)
    return xmin, xmax, ymin-pad, ymax+pad


def draw_force_overlay(x, gt, pred, break_after=None):
    xmin, xmax, ymin, ymax = curve_limits(x, gt, pred)
    origin = imgui.GetCursorScreenPos()
    width = max(300., min(650., float(imgui.GetContentRegionAvail()[0])))
    height = 235.
    imgui.InvisibleButton('##force_curve_overlay', (width, height))
    draw = imgui.GetWindowDrawList()
    left, right = origin[0]+62., origin[0]+width-12.
    top, bottom = origin[1]+12., origin[1]+height-30.
    color = lambda rgba: imgui.GetColorU32(rgba)
    grid = color((.35,.35,.35,1.))
    text = color((.9,.9,.9,1.))
    gt_color = color((1.,.65,.12,1.))
    pred_color = color((.15,.8,1.,1.))
    draw.AddRect((left,top),(right,bottom),grid)
    for fraction in (0., .5, 1.):
        yy = bottom-fraction*(bottom-top)
        xx = left+fraction*(right-left)
        draw.AddLine((left,yy),(right,yy),grid)
        draw.AddText((origin[0],yy-6.),text,f'{ymin+fraction*(ymax-ymin):.3g}')
        draw.AddText((xx-8.,bottom+5.),text,f'{xmin+fraction*(xmax-xmin):.3g}')
    x = np.asarray(x)
    for values, line_color in ((gt,gt_color),(pred,pred_color)):
        px = left+(x-xmin)/(xmax-xmin)*(right-left)
        py = bottom-(np.asarray(values)-ymin)/(ymax-ymin)*(bottom-top)
        for i in range(len(x)-1):
            if break_after is not None and i == break_after-1:
                continue
            draw.AddLine((float(px[i]),float(py[i])),(float(px[i+1]),float(py[i+1])),line_color,2.)
    imgui.Text('Displacement (mm); Force (N). GT: orange; prediction: cyan.')
    if break_after is not None:
        imgui.Text('Loss uses two separated frame ranges; no line is drawn across the gap.')
