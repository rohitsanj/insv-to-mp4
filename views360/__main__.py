"""python -m views360 [doctor | stitch ... | CAMERA_FILE] [options]"""
import sys


def main():
    args = sys.argv[1:]
    if args[:1] == ["doctor"]:
        from .system import doctor
        sys.exit(0 if doctor() else 1)
    if args[:1] == ["stitch"]:
        from . import stitch360
        sys.argv = ["360views stitch"] + args[1:]
        return stitch360.main()
    from .server import main as serve
    serve(args)


if __name__ == "__main__":
    main()
