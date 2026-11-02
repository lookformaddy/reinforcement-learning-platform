.PHONY: test demo lint

test:
	pytest -q

demo:
	python3 demo.py

lint:
	python3 -m py_compile src/rl_platform/*.py demo.py
