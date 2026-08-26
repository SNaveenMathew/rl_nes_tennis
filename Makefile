.PHONY: test compile

test:
	python -m unittest discover -s tests -v

compile:
	python -m compileall -q tennis_rl tools tests
