"""Setup file for scikinC

Original template at:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='scikinC',  # Required
    version="0.2.8",  # Required
    description='A converter for scikit learn and keras to hardcoded C function',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/landerlini/scikinC',  # Optional
    author='Lucio Anderlini',  # Optional
    author_email='Lucio.Anderlini@fi.infn.it',  # Optional
    classifiers=[  # Optional
        # Project maturity
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
				'Intended Audience :: Science/Research', 
				'Topic :: Software Development :: Code Generators', 

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='deployment, header-only, plain-C',  # Optional

    #package_dir={'': 'scikinC'},  # Optional

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',

    install_requires=['numpy', 'scipy', 'scikit-learn'],  # Optional

    extras_require={  # Optional
        'keras': ['tensorflow', 'keras'],
        'fql': ['tensorflow', 'keras', 'fastquantilelayer'],
    },

    entry_points={  # Optional
        'console_scripts': [
            'scikinC=scikinC.__main__:main',
        ],
    },
)
