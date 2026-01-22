FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci || npm install

COPY . .

ARG VITE_API_BASE=/api
ARG VITE_WS_URL=/ws
ENV VITE_API_BASE=${VITE_API_BASE}
ENV VITE_WS_URL=${VITE_WS_URL}

RUN npm run build

FROM nginx:alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
