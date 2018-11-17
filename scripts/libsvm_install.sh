cd ./dojo/svm
rm -r ./libsvm
git clone https://github.com/cjlin1/libsvm
cd ./libsvm
make
touch __init__.py
cd ./python
make
touch __init__.py
