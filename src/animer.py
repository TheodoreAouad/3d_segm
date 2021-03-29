import vpython as vp


def draw(fn):
    """
    Draws a new scene from the given scene

    Args:
        fn: (todo): write your description
    """
    scene = vp.canvas()
    fn()
    return scene
