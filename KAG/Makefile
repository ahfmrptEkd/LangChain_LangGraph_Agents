.PHONY: help start stop status logs clean

# 기본 타겟
help: ## 사용 가능한 명령어를 보여줍니다
	@echo "🔧 Neo4j 관리 명령어:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

start: ## Neo4j Docker 컨테이너를 시작합니다
	@echo "🚀 Neo4j Docker 컨테이너를 시작합니다..."
	docker-compose up -d
	@echo "✅ Neo4j가 성공적으로 시작되었습니다!"
	@echo ""
	@echo "📊 Neo4j 브라우저: http://localhost:7474"
	@echo "🔑 기본 로그인 정보:"
	@echo "   - 사용자명: neo4j"
	@echo "   - 비밀번호: password"

stop: ## Neo4j Docker 컨테이너를 중지
	@echo "🛑 Neo4j Docker 컨테이너를 중지합니다..."
	docker-compose down
	@echo "✅ Neo4j가 중지되었습니다."

status: ## 컨테이너 상태를 확인
	@echo "🔍 Neo4j 컨테이너 상태:"
	docker-compose ps

logs: ## Neo4j 로그를 확인
	@echo "📋 Neo4j 로그:"
	docker-compose logs neo4j

clean: ## 모든 컨테이너와 볼륨을 삭제
	@echo "🧹 모든 Neo4j 데이터를 삭제합니다..."
	docker-compose down -v
	@echo "✅ 모든 데이터가 삭제되었습니다."

restart: stop start ## Neo4j를 재시작
