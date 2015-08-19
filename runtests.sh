echo "Testing on python 2.7"
nosetests-2.7 -v --with-coverage --cover-package=proxalgs --cover-html

echo "Testing on python 3.4"
nosetests-3.4 -v --with-coverage --cover-package=proxalgs --cover-html

echo "Opening coverage results"
open cover/index.html
