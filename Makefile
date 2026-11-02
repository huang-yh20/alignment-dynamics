.PHONY: analysis test verify

analysis:
	python -m alignment_dynamics.analyze
	python -m alignment_dynamics.table3
	python -m alignment_dynamics.plot --figure all

test:
	python -m unittest discover -s tests -v

verify:
	python -m alignment_dynamics.verify --data
