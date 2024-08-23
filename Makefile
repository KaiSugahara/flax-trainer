export
YOUR_UID := ${shell id -u}
YOUR_GID := ${shell id -g}

dev:
	docker compose build --no-cache
	devcontainer open .

clean:
	docker compose down --rmi all --volumes