"""Red GUI theme while training records are disabled; restore the user's colors."""
from polyscope import imgui

_original = None
_red_active = False


def set_logging_theme(enabled):
    global _original, _red_active
    colors = imgui.GetStyle().Colors
    if _original is None:
        names = ['WindowBg','TitleBg','TitleBgActive','FrameBg','FrameBgHovered',
                 'FrameBgActive','Button','ButtonHovered','ButtonActive','CheckMark',
                 'SliderGrab','SliderGrabActive','Header','HeaderHovered','HeaderActive']
        _original = {getattr(imgui, 'ImGuiCol_'+name):tuple(colors[getattr(imgui, 'ImGuiCol_'+name)]) for name in names}
    if enabled:
        if _red_active:
            for index, color in _original.items():
                colors[index] = color
            _red_active = False
        return
    if not _red_active:
        for index in _original:
            colors[index] = (.50, .10, .10, 1.)
        colors[imgui.ImGuiCol_WindowBg] = (.23, .045, .045, .98)
        colors[imgui.ImGuiCol_FrameBg] = (.32, .065, .065, 1.)
        colors[imgui.ImGuiCol_TitleBgActive] = (.55, .08, .08, 1.)
        _red_active = True
