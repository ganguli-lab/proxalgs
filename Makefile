all:
	python setup.py install

develop:
	python setup.py develop

test:
	py.test -v --cov=proxalgs --cov-report=html tests

lint:
	flake8 descent/

clean:
	rm -rf htmlcov/
	rm -rf proxalgs.egg-info
	rm -f proxalgs/*.pyc
	rm -rf proxalgs/__pycache__

upload:
	python setup.py sdist upload
