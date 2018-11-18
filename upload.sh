sudo .
pipenv run python ./setup.py sdist bdist_wheel
pipenv run twine upload ./dist/*
rm -r ./dist
rm -r ./build
rm -r ./PyDojoML.egg-info