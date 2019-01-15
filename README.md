# Dojo
![Dojo_logo](./img/logo_transparent.png)

Dojo is a Machine Learning library for Python

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

* [Python](https://www.python.org/) - The Programming Language used.
* [Pipenv](https://github.com/pypa/pipenv) - Dependency and Virtual Environment Management

***Download for Mac OSX using Homebrew***
```
brew install python
brew install pipenv
```

### Installing for development

A step by step series of examples that tell you how to get a development env running

1) Since we are using the **Python** programming language as a main language, you will need to download it.
You can do so from the official **Python** [website](https://www.python.org/).

2) Once you have **Python** up and running we then need to setup our development env. For that
we are using **Pipenv**. You will need to install it. Check out [these](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv) instructions to see how is done.

3) Now, that you have the prerequisites the only part left is too install all the other **Pyhton** packages
that **Dojo** depends on. To do run the following:
    ```
    pipenv install --dev
    ```
    The `--dev` tag is used in order **Pipenv** to know to install also the packages that are used in the
    package development process.

### Installing for use

If you plan just to use **Dojo** as a Machine Learning library you can install it using **pip** like so:
```
pip install pydojoml
``` 

## Running the tests

Coming soon...

### Break down into end to end tests

Coming soon...

### And coding style tests

Coming soon...

## Built With

* [NumPy](http://www.numpy.org/) - Fundamental package for scientific computing with Python
* [SciPy](http://www.scipy.org/) - Package that provides many user-friendly and efficient numerical routines
* [Matplotlib](http://www.matplotlib.org/) - Python 2D plotting library
* [progressbar](https://pypi.org/project/progressbar/) - Text progress bar library for Python
* [terminaltables](https://pypi.org/project/terminaltables/) - Easily draw tables in terminal/console applications

## Contributing

Please read [CONTRIBUTING.md](https://github.com/VIVelev/PyDojoML/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

For the versions available, see the [tags on this repository](https://github.com/VIVelev/PyDojoML/tags). 

## Authors

* **Victor Velev** - *Initial work* - [VIVelev](https://github.com/VIVelev)

See also the list of [contributors](https://github.com/VIVelev/PyDojoML/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* **Eric Jones and Travis Oliphant and Pearu Peterson and others** for writing such great packages - the [SciPy](http://www.scipy.org/) ecosystem.
* **Nilton Volpato** for writing [progressbar](https://pypi.org/project/progressbar/)
* **Robpol86** for writing [terminaltables](https://pypi.org/project/terminaltables/)
