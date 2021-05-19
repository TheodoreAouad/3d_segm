import vpython as vp


def draw(fn):
    scene = vp.canvas()
    fn()
    return scene
