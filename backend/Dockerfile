# Stage 1: Build
FROM golang:latest AS builder

WORKDIR /app

COPY go.mod ./
# Se hai dipendenze esterne in futuro, decommenta la riga sotto
# COPY go.sum ./
# RUN go mod download

COPY . .

# Compilazione statica
RUN CGO_ENABLED=0 GOOS=linux go build -o orchestrator .

# Stage 2: Runtime
FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/orchestrator .

CMD ["./orchestrator"]
