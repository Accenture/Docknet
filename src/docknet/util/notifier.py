import sys


# These codes change the font color when printed inside ANSI compatible
# terminals; more info on ANSI escape codes here:
# https://en.wikipedia.org/wiki/ANSI_escape_code
ANSI_CODES = {'ERROR': '\033[91m\033[1m',
              'WARN': '\033[93m\033[1m',
              'INFO': '\033[92m\033[1m',
              'off': '\033[0m\033[21m'}


def format_string(string: str, *args) -> str:
    """
    Format & return a string
    """
    return string if not args else string.format(*args)


class Notifier(object):
    """
    A small object for printing progress info to console
    """
    def __init__(self, debug: bool = False, force: bool = False):
        """Initialize the notifier class
          :param debug (bool): whether to print debug messages or not
          :param force (bool): force colored printing, even if the output
          stream
            is not a terminal
        """
        self._debug = debug
        self.cc = (ANSI_CODES if force or sys.stdout.isatty()
                   else {'ERROR': '', 'WARN': '', 'INFO': '', 'off': ''})

    def _msg(self, header: str, string: str, *args, **kwargs):
        print(header + format_string(string, *args) + self.cc['off'],
              flush=True, **kwargs)

    def error(self, string: str, *args, **kwargs):
        """
        Print an error message
        """
        self._msg(self.cc['ERROR'], string, *args, **kwargs)

    def warn(self, string: str, *args, **kwargs):
        """
        Print a warning message
        """
        self._msg(self.cc['WARN'], string, *args, **kwargs)

    def info(self, string: str, *args, **kwargs):
        """
        Print an information message
        """
        self._msg(self.cc['INFO'], string, *args, **kwargs)

    def debug(self, string: str, *args):
        """
        Print a debug message
        """
        if self._debug:
            self._msg('', string, *args)
