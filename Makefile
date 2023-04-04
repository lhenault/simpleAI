.PHONY: install-dev precommit

install-dev:
	poetry install --with dev
	poetry run pre-commit install
