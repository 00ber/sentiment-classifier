.PHONY: build-backend
build-backend: ## Build backend services
	docker build -f Dockerfile.backend -t sentiment-classification-backend ./backend

.PHONY: build-frontend
build-frontend: ## Build frontend services
	docker build -f Dockerfile.frontend -t sentiment-classification-frontend ./frontend

.PHONY: up-prod
up-prod: ## Run backend services prod
	docker build -f Dockerfile.backend -t sentiment-classification-backend ./backend
	docker build -f Dockerfile.frontend -t sentiment-classification-frontend ./frontend
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d 

.PHONY: up-dev
up-dev: ## Run backend services dev
	docker build -f Dockerfile.backend -t sentiment-classification-backend ./backend
	docker build -f Dockerfile.frontend -t sentiment-classification-frontend ./frontend
	docker-compose up -d 
