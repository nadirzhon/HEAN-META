FROM node:20-alpine AS builder

WORKDIR /app

# Build arguments so Vite receives the proper environment variables.
ARG VITE_API_BASE=/api
ARG VITE_WS_URL=/ws
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_WS_URL=$VITE_WS_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.25-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
