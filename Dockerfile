FROM lmsysorg/sglang:latest

COPY start.sh /start.sh
RUN chmod +x /start.sh

# OpenAI-compatible HTTP API
EXPOSE 30000

CMD ["/start.sh"]
