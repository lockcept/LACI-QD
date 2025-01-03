"""
Module containing utility functions and classes.
"""


class AverageMeter:
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        """
        Update the average meter with a new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Docdict(dict):
    """
    dot.notation access to dictionary attributes.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Attribute {name} not found") from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __getstate__(self):
        """
        Ensures compatibility with pickle during serialization.
        """
        return self.__dict__.copy()

    def __setstate__(self, state):
        """
        Ensures compatibility with pickle during deserialization.
        """
        self.__dict__.update(state)
