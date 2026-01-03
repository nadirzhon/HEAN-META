FROM nginx:alpine

# Copy website files to nginx html directory
COPY . /usr/share/nginx/html

# Copy custom nginx configuration (optional, for better performance)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

