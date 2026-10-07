"""AMRL-Fast benchmark extension package.

The package deliberately avoids importing the optional MuJoCo/PyTorch stack at
package-import time, so analysis utilities can be tested independently on a
machine that is only preparing the Linux experiment bundle.
"""

__all__: list[str] = []
