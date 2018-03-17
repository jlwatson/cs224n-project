import os
from subprocess import call

def strip_substring(s, subs, suffix=False):
    if not suffix:
        idx = s.find(subs)
    else:
        idx = s.find(subs, len(s) - len(subs))
    return s[:idx] + s[idx + len(subs):]

class FileProcessing:

    DEFAULT_TEMP_FILENAME = 'temp.txt'

    def __init__(self, raw_filename):
        with open(raw_filename, 'r') as f:
            self.lines = [l.strip() for l in f.readlines()]

        self.temp_filename = self.DEFAULT_TEMP_FILENAME
        self.temp_file = open(self.temp_filename, 'w')
        self.flush()

        if os.path.isfile('/usr/local/bin/sublime'):
            call(['sublime', self.temp_file.name])

        self.prev = None

    def flush(self):
        self.temp_file.seek(0)
        self.temp_file.truncate()
        self.temp_file.writelines(
            [l + '\n' for l in self.lines]
        )
        self.temp_file.flush()
        os.fsync(self.temp_file.fileno())

        if os.path.isfile('/usr/local/bin/sublime'):
            call(['sublime'])

    def output(self, final_filename):
        with open(final_filename, 'w+') as final:
            final.writelines(
                [l + '\n' for l in self.lines]
            )

    def close(self):
        self.temp_file.close()
        os.remove(self.temp_filename)

    def _save(self):
        self.prev = {
            'lines': self.lines,
            'prev': self.prev,
        }

    def undo(self):
        if self.prev is not None:
            self.lines = self.prev['lines']
            self.prev = self.prev['prev']
            self.flush()
        else:
            print("Nothing to undo.")

    def operation(self, op):
        self._save()
        self.lines = op(self.lines)
        self.flush()

    def remove_empty(self):
        self.operation(
            lambda lines: [l for l in lines if l != ""]
        )

    def remove_contains(self, a):
        self.operation(
            lambda lines: [l for l in lines if a not in l]
        )

    def remove_exact(self, a):
        self.operation(
            lambda lines: [l for l in lines if a != l]
        )

    def remove_prefix(self, a):
        self.operation(
            lambda lines: [l for l in lines if not l.startswith(a)]
        )

    def remove_suffix(self, a):
        self.operation(
            lambda lines: [l for l in lines if not l.endswith(a)]
        )

    def remove_range(self, start, end):
        self.operation(
            lambda lines: lines[:start] + lines[end:]
        )

    def strip_contains(self, a):
        self.operation(
            lambda lines: [strip_substring(l, a) if a in l else l for l in lines]
        )

    def strip_prefix(self, a):
        self.operation(
            lambda lines: [strip_substring(l, a) if l.startswith(a) else l for l in lines]
        )

    def strip_suffix(self, a):
        self.operation(
            lambda lines: [strip_substring(l, a, True) if l.endswith(a) else l for l in lines]
        )

    def remove_predicate(self, pred):
        self.operation(
            lambda lines: [l for l in lines if pred(l)]
        )

