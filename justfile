IMAGE := "llamaqueue"

build:
	docker build -t {{IMAGE}} .

test:
	pip install -r requirements-test.txt
	pytest test_proxy.py -v
