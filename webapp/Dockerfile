FROM python:3.11-slim

WORKDIR /app

# Only the two things inference actually needs at runtime: the four project
# source files (real forward-pass code, imported by file path) and the
# already-trained weight JSON files. Nothing else from the monorepo.
COPY webapp/requirements.txt ./webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

COPY projects/ ./projects/
COPY portfolio-website/model_weights/ ./portfolio-website/model_weights/
COPY webapp/ ./webapp/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces' Docker SDK expects the app to listen on port 7860.
ENV BACKEND_URL=http://localhost:8000
EXPOSE 7860

WORKDIR /app/webapp
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
