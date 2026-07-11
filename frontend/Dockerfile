# Stage 1: Build static React assets
FROM node:20-alpine as build
WORKDIR /frontend

# Set default API base URL to /api to leverage Nginx reverse-proxy
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Copy package configuration
COPY package.json ./

# Install dependencies
RUN npm install

# Copy source code, configurations, and public directory
COPY tsconfig.json tsconfig.node.json vite.config.ts postcss.config.js tailwind.config.js index.html ./
COPY src ./src
COPY public ./public

# Build the production bundle
RUN npm run build

# Stage 2: Serve using Nginx
FROM nginx:1.25-alpine

# Copy built assets from builder stage
COPY --from=build /frontend/dist /usr/share/nginx/html

# Copy custom Nginx configuration for routing and reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
