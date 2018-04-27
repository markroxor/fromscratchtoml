NOTEBOOK_HTML_DIR="docs/project/static/notebooks"
for f in docs/notebooks/*.ipynb
do
    eval "jupyter nbconvert $f --output-dir $NOTEBOOK_HTML_DIR"
done
