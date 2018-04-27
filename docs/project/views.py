from flask import render_template, Markup
from flask_assets import Environment, Bundle
import os
from app import app
import os.path
from werkzeug.contrib.cache import SimpleCache
import markdown

cache = SimpleCache()

NOTEBOOK_HTML_DIR = os.path.dirname(os.path.realpath(__file__)) + "/static/notebooks"
NOTEBOOK_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/notebooks"


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/install/')
def install():
    install_content = get_markuped("install", "static/mds/INSTALL.md")
    return render_template('install.html', **locals())


@app.route('/showroom/')
def showroom():
    notebooks = get_notebooks()
    return render_template('showroom.html', examples=notebooks)


# assets
assets = Environment(app)

scss = Bundle('stylesheets/main.scss', filters='pyscss', output='gen/scss.css')
all_css = Bundle('vendor/*.css', scss, filters='cssmin', output="gen/all.css")
assets.register('css_all', all_css)

js = Bundle(
    'vendor/jquery-3.1.1.min.js',
    'vendor/jquery.timeago.js',
    'vendor/bootstrap.min.js',
    'vendor/showdown.min.js',
    'javascripts/*.js',
    filters='jsmin', output='gen/packed.js'
)
assets.register('js_all', js)


def get_markuped(key, filename):
    content = cache.get(key)
    if content is None:
        with open(os.path.join(app.root_path, filename)) as f:
            content = ''.join(f.readlines())
        content = Markup(markdown.markdown(content))
        cache.set(key, content, timeout=0)
    return content


# utils
def get_abstract(fname):
    import json
    import os
    import markdown

    try:
        with open(fname) as f:
            js = json.load(f)

        if 'worksheets' in js:
            if len(js['worksheets']) > 0:
                if js['worksheets'][0]['cells'] is not None:
                    cells = js['worksheets'][0]['cells']
        else:
            if 'cells' in js:
                cells = js['cells']

        for cell in cells:
            if cell['cell_type'] == 'heading' or cell['cell_type'] == 'markdown':
                return markdown.markdown(''.join(cell['source'][0]).replace('#', ''))
    except Exception as e:
        print(e, "\n")
        pass

    return os.path.basename(fname)


def get_notebooks():
    notebooks = []
    # ipython nbconvert --to FORMAT notebook.ipynb
    rel_path = "/fromscratchtoml/static/notebooks/"
    # print(os.)
    for _file in os.listdir(NOTEBOOK_HTML_DIR):
        if _file.endswith(".html"):
            notebook_url = _file
            notebook_image = notebook_url.replace('.html', '.png')
            if not os.path.isfile(os.path.abspath(os.path.join(os.path.realpath(__file__), '../static/notebooks',
            notebook_image))):
                notebook_image = "default.png"

            notebook_title = _file[0:-5].replace('_', ' ')
            notebook_abstract = get_abstract(os.path.abspath(os.path.join(os.path.realpath(__file__), '../../notebooks',
            _file.replace('.html', '.ipynb'))))
            notebooks.append({
                'url': rel_path + notebook_url,
                'image': rel_path + notebook_image,
                'title': notebook_title,
                'abstract': notebook_abstract})

    return notebooks
