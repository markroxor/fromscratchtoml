NOTEBOOK_HTML_DIR="project/static/notebooks"
for f in notebooks/*.ipynb
do
    eval "jupyter nbconvert $f --output-dir $NOTEBOOK_HTML_DIR"
done
