import sys
import unittest
import importlib
import pathlib


def main():
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    if not args:
        args = ['tests']

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for arg in args:
        path = pathlib.Path(arg)
        if path.is_dir():
            for file in path.rglob('test*.py'):
                mod_name = '.'.join(file.with_suffix('').parts)
                mod = importlib.import_module(mod_name)
                for name in dir(mod):
                    if name.startswith('test') and callable(getattr(mod, name)):
                        suite.addTest(unittest.FunctionTestCase(getattr(mod, name)))
        elif path.is_file():
            mod_name = '.'.join(path.with_suffix('').parts)
            mod = importlib.import_module(mod_name)
            for name in dir(mod):
                if name.startswith('test') and callable(getattr(mod, name)):
                    suite.addTest(unittest.FunctionTestCase(getattr(mod, name)))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
