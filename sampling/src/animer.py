import vpython as vp


def draw(fn):
    """
    Draws a scene using the given function fn.

    Args:
        fn: write your description
    """
    scene = vp.canvas()
    fn()
    return scene
