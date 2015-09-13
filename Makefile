all:
	python setup.py install

develop:
	python setup.py develop

test:
	py.test -v --cov=proxalgs --cov-report=html tests

clean:
	rm -rf htmlcov/
	rm -rf proxalgs.egg-info
	rm -f proxalgs/*.pyc
	rm -rf proxalgs/__pycache__
