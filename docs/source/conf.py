# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath('../../..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../mowl'))
sys.path.insert(0, os.path.abspath('../../gateway/src/main/scala/org'))
# -- Project information

project = 'MOWL'
copyright = '2023, BORG'
author = 'BORG'

release = '0.3.0'
version = '0.3.0'
# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
#    'IPython.sphinxext.ipython_console_highlighting'

    # Matplotlib
    #'matplotlib.sphinxext.only_directives',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    
    #'numpydoc'
]

doctest_global_setup = """
import jpype
import jpype.imports
import os
dirname = os.path.dirname("../mowl/")
jars_dir = os.path.join(dirname, "lib/")
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'
if not jpype.isJVMStarted():
    jpype.startJVM(
           jpype.getDefaultJVMPath(), "-ea",
	   "-Xmx10g",
	   "-Djava.class.path=" + jars,
	   convertStrings=False)
"""

todo_include_todos = True

examples_dirs = [
    '../../examples/'
]

gallery_dirs = [
    'examples/']

sphinx_gallery_conf = {
    'examples_dirs': examples_dirs,   # path to your example scripts
    'gallery_dirs': gallery_dirs,  # path to where to save gallery generated output
    'filename_pattern': 'none',
    "within_subsection_order": FileNameSortKey,
    "run_stale_examples": True,
    "abort_on_example_error": False,
    #"plot_gallery": False,
    "show_memory": True,
}

autodoc_member_order = 'bysource'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'pykeen': ('https://pykeen.readthedocs.io/en/latest/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'gensim': ('https://radimrehurek.com/gensim/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

#html_logo = 'mowl-logo.jpg'
html_logo = 'mowl_white_background_colors_2048x2048px.png'
#html_logo = 'mowl_black_background_colors_2048x2048px.png'


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

autodoc_mock_imports = ['jpype', 'owlready2', 'rdflib', 'networkx', 'node2vec', 'matplotlib']

#autodoc_mock_imports = ['org', 'uk', 'java', 'numpy', 'jpype', 'de', 'pandas', 'scipy', 'sklearn', 'owlready2', 'gensim', 'torch', 'rdflib', 'networkx', 'pykeen', 'node2vec', 'matplotlib', 'tqdm', 'click']


import mowl
mowl.init_jvm("4g")
