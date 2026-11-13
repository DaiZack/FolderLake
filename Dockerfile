FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 8770
# Registry (history + settings) lives in ~/.folderlake — mount a volume
# over /root/.folderlake to keep it across container rebuilds.
CMD ["folderlake", "app", "--host", "0.0.0.0", "--port", "8770"]
