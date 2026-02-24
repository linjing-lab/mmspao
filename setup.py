from setuptools import setup
from mmspao import __version__

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
      name='mmspao',  # pkg_name
      packages=['mmspao',],
      version=__version__,  # version number
      description="Multi-modal spatial omics integration with views and combinations.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/mmspao',
      download_url='https://github.com/linjing-lab/mmspao/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/mmspao/tree/main/mmspao/',
            'Tracker': 'https://github.com/linjing-lab/mmspao/issues',
      },
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=[
            'numpy>=1.26.4', # 'numpy==1.26.4'
            'scanpy>=1.10.4',
            'scikit-learn>=1.7.1',
            'scipy>=1.15.3',
            'tqdm>=4.67.1',
      ],
      # extras_require=[]
)