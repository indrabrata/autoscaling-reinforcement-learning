run:
	python -m model.train

compose-dev:
	docker compose -f compose.yaml up -d

compose-dev-down:
	docker compose -f compose.yaml down
