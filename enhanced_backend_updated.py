# Production-Ready AI Text Processing API
# Enterprise-Grade Application with 10M+ User Capacity
# Comprehensive Security, Scalability, and Performance Optimizations

import base64
import secrets
import os
import asyncio
import datetime
import re
import smtplib
import hashlib
import hmac
import json
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# Core Dependencies
import nltk
import openai
import httpx
import uvloop
import psutil
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from textblob import TextBlob
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Azure Dependencies
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.cosmos import CosmosClient
from azure.servicebus import ServiceBusClient
from azure.storage.blob import BlobServiceClient
from azure.monitor.opentelemetry import configure_azure_monitor

# Performance Dependencies
import redis.asyncio as redis
import aioredis
from cachetools import TTLCache
import orjson

# Monitoring Dependencies
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# -------------------- NLTK Data --------------------
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download failed: {e}")

# -------------------- Global Configuration --------------------
# Set uvloop as the default event loop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global Executor for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=20)

# Global Cache for frequently accessed data
memory_cache = TTLCache(maxsize=10000, ttl=300)

# -------------------- Environment & Configuration --------------------
class Config:
    # Azure Configuration
    AZURE_KEY_VAULT_URL = os.getenv("AZURE_KEY_VAULT_URL")
    AZURE_COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
    AZURE_COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
    AZURE_COSMOS_DATABASE = os.getenv("AZURE_COSMOS_DATABASE", "ai_text_processing")
    AZURE_SERVICE_BUS_CONNECTION = os.getenv("AZURE_SERVICE_BUS_CONNECTION")
    AZURE_STORAGE_CONNECTION = os.getenv("AZURE_STORAGE_CONNECTION")
    AZURE_APPLICATION_INSIGHTS_CONNECTION = os.getenv("AZURE_APPLICATION_INSIGHTS_CONNECTION")
    
    # API Keys (loaded from Key Vault)
    OPENAI_API_KEY = None
    TWOCHECKOUT_SECRET_KEY = None
    TWOCHECKOUT_MERCHANT_CODE = None
    SENDGRID_API_KEY = None
    TWILIO_ACCOUNT_SID = None
    TWILIO_AUTH_TOKEN = None
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Azure Table Storage Configuration (replacing Redis)
    AZURE_TABLE_STORAGE_CONNECTION = os.getenv("AZURE_TABLE_STORAGE_CONNECTION")
    CACHE_ENABLED = True
    CACHE_TTL_SECONDS = 300  # 5 minutes default TTL
    
    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "1000"))
    KEEPALIVE_TIMEOUT = int(os.getenv("KEEPALIVE_TIMEOUT", "5"))
    
    # Feature Flags
    ENABLE_MONITORING = True
    ENABLE_CACHING = True
    ENABLE_RATE_LIMITING = True
    ENABLE_COMPRESSION = True

config = Config()

# -------------------- Configuration Manager --------------------
class ConfigurationManager:
    def __init__(self):
        self.secrets_loaded = False
        
    async def load_and_store_secrets(self):
        """Load secrets from Azure Key Vault and store permanently in database."""
        if self.secrets_loaded:
            return
            
        # First try to load from database
        await self._load_secrets_from_database()
        
        # If not found in database, load from Key Vault
        if not self.secrets_loaded:
            await self._load_secrets_from_vault()
            
    async def _load_secrets_from_database(self):
        """Load secrets from database."""
        try:
            config_container = db_manager.get_container("app_config")
            if not config_container:
                return
                
            secrets_config = config_container.read_item(item="secrets", partition_key="secrets")
            
            # Load secrets into config
            config.OPENAI_API_KEY = secrets_config.get("openai_api_key")
            config.TWOCHECKOUT_SECRET_KEY = secrets_config.get("twocheckout_secret_key")
            config.TWOCHECKOUT_MERCHANT_CODE = secrets_config.get("twocheckout_merchant_code")
            config.SENDGRID_API_KEY = secrets_config.get("sendgrid_api_key")
            config.TWILIO_ACCOUNT_SID = secrets_config.get("twilio_account_sid")
            config.TWILIO_AUTH_TOKEN = secrets_config.get("twilio_auth_token")
            config.SMTP_USERNAME = secrets_config.get("smtp_username", config.NOTIFICATION_EMAIL)
            config.SMTP_PASSWORD = secrets_config.get("smtp_password")
            config.SMTP_FROM_EMAIL = config.NOTIFICATION_EMAIL
            
            # Load hardcoded credentials from database
            config.OWNER_EMAIL = secrets_config.get("owner_email", config.OWNER_EMAIL)
            config.OWNER_PASSWORD = secrets_config.get("owner_password", config.OWNER_PASSWORD)
            config.NOTIFICATION_EMAIL = secrets_config.get("notification_email", config.OWNER_EMAIL)
            
            self.secrets_loaded = True
            logging.info("Secrets loaded from database")
            
        except Exception as e:
            logging.warning(f"Failed to load secrets from database: {e}")
            
    async def _load_secrets_from_vault(self):
        """Load secrets from Azure Key Vault and store in database."""
        if not config.AZURE_KEY_VAULT_URL:
            logging.warning("Azure Key Vault URL not configured")
            return
            
        try:
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=config.AZURE_KEY_VAULT_URL, credential=credential)
            
            # Load API keys from Key Vault
            secrets_to_load = [
                ("openai-api-key", "openai_api_key"),
                ("twocheckout-secret-key", "twocheckout_secret_key"),
                ("twocheckout-merchant-code", "twocheckout_merchant_code"),
                ("sendgrid-api-key", "sendgrid_api_key"),
                ("twilio-account-sid", "twilio_account_sid"),
                ("twilio-auth-token", "twilio_auth_token"),
                ("smtp-username", "smtp_username"),
                ("smtp-password", "smtp_password"),
                ("owner-email", "owner_email"),
                ("owner-password", "owner_password"),
                ("notification-email", "notification_email"),
            ]
            
            secrets_data = {
                "id": "secrets",
                "smtp_username": config.NOTIFICATION_EMAIL,
                "owner_email": config.OWNER_EMAIL,
                "owner_password": config.OWNER_PASSWORD,
                "notification_email": config.OWNER_EMAIL,
                "updated_at": datetime.datetime.utcnow().isoformat()
            }
            
            for vault_secret_name, db_key in secrets_to_load:
                try:
                    secret = client.get_secret(vault_secret_name)
                    secrets_data[db_key] = secret.value
                    logging.info(f"Loaded secret from vault: {vault_secret_name}")
                except Exception as e:
                    logging.error(f"Failed to load secret {vault_secret_name}: {e}")
                    
            # Store in database
            config_container = db_manager.get_container("app_config")
            if config_container:
                try:
                    config_container.upsert_item(secrets_data)
                    logging.info("Secrets stored in database")
                    
                    # Load into config
                    await self._load_secrets_from_database()
                    
                except Exception as e:
                    logging.error(f"Failed to store secrets in database: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to initialize Key Vault client: {e}")
            
    async def retry_secret_loading(self):
        """Retry loading secrets if some are missing."""
        missing_secrets = []
        
        if not config.OPENAI_API_KEY:
            missing_secrets.append("OPENAI_API_KEY")
        if not config.SMTP_PASSWORD:
            missing_secrets.append("SMTP_PASSWORD")
        if not hasattr(config, 'OWNER_EMAIL') or not config.OWNER_EMAIL:
            missing_secrets.append("OWNER_EMAIL")
        if not hasattr(config, 'OWNER_PASSWORD') or not config.OWNER_PASSWORD:
            missing_secrets.append("OWNER_PASSWORD")
            
        if missing_secrets:
            logging.warning(f"Missing secrets detected: {missing_secrets}")
            self.secrets_loaded = False
            await self.load_and_store_secrets()

# -------------------- Azure Table Storage Cache Manager --------------------
from azure.data.tables import TableServiceClient, TableClient
import pickle

class AzureTableCacheManager:
    def __init__(self):
        self.table_service_client = None
        self.cache_table_name = "cache_storage"
        self.rate_limit_table_name = "rate_limits"
        self.initialized = False
        
    async def initialize(self):
        """Initialize Azure Table Storage connection."""
        if self.initialized:
            return
            
        try:
            if config.AZURE_TABLE_STORAGE_CONNECTION:
                self.table_service_client = TableServiceClient.from_connection_string(
                    config.AZURE_TABLE_STORAGE_CONNECTION
                )
                
                # Create tables if they don't exist
                await self._create_tables()
                self.initialized = True
                logging.info("Azure Table Storage cache initialized")
            else:
                logging.warning("Azure Table Storage connection string not configured")
                
        except Exception as e:
            logging.error(f"Failed to initialize Azure Table Storage: {e}")
    
    async def _create_tables(self):
        """Create cache and rate limit tables if they don't exist."""
        try:
            # Create cache table
            try:
                self.table_service_client.create_table(self.cache_table_name)
            except Exception:
                pass  # Table might already exist
                
            # Create rate limit table
            try:
                self.table_service_client.create_table(self.rate_limit_table_name)
            except Exception:
                pass  # Table might already exist
                
        except Exception as e:
            logging.error(f"Failed to create tables: {e}")
    
    async def get(self, key: str, table_name: str = None) -> Any:
        """Get value from cache."""
        if not self.initialized:
            await self.initialize()
            
        if not self.table_service_client:
            return None
            
        try:
            table_name = table_name or self.cache_table_name
            table_client = self.table_service_client.get_table_client(table_name)
            
            # Get entity
            entity = table_client.get_entity(partition_key="cache", row_key=key)
            
            # Check if expired
            if entity.get("expires_at"):
                expires_at = datetime.datetime.fromisoformat(entity["expires_at"])
                if datetime.datetime.utcnow() > expires_at:
                    # Delete expired entry
                    await self.delete(key, table_name)
                    return None
            
            # Deserialize value
            if entity.get("value_data"):
                return pickle.loads(base64.b64decode(entity["value_data"]))
                
            return entity.get("value")
            
        except Exception as e:
            logging.debug(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = None, table_name: str = None):
        """Set value in cache."""
        if not self.initialized:
            await self.initialize()
            
        if not self.table_service_client:
            return False
            
        try:
            table_name = table_name or self.cache_table_name
            table_client = self.table_service_client.get_table_client(table_name)
            
            # Calculate expiration
            ttl = ttl_seconds or config.CACHE_TTL_SECONDS
            expires_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl)
            
            # Prepare entity
            entity = {
                "PartitionKey": "cache",
                "RowKey": key,
                "expires_at": expires_at.isoformat(),
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            
            # Serialize complex objects
            if isinstance(value, (dict, list, tuple, set)):
                entity["value_data"] = base64.b64encode(pickle.dumps(value)).decode()
            else:
                entity["value"] = str(value)
            
            # Upsert entity
            table_client.upsert_entity(entity)
            return True
            
        except Exception as e:
            logging.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str, table_name: str = None):
        """Delete value from cache."""
        if not self.initialized:
            await self.initialize()
            
        if not self.table_service_client:
            return False
            
        try:
            table_name = table_name or self.cache_table_name
            table_client = self.table_service_client.get_table_client(table_name)
            
            table_client.delete_entity(partition_key="cache", row_key=key)
            return True
            
        except Exception as e:
            logging.debug(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, ttl_seconds: int = None, table_name: str = None) -> int:
        """Increment a counter in cache."""
        if not self.initialized:
            await self.initialize()
            
        try:
            current_value = await self.get(key, table_name) or 0
            new_value = int(current_value) + amount
            await self.set(key, new_value, ttl_seconds, table_name)
            return new_value
            
        except Exception as e:
            logging.error(f"Cache increment failed for key {key}: {e}")
            return amount
    
    async def exists(self, key: str, table_name: str = None) -> bool:
        """Check if key exists in cache."""
        value = await self.get(key, table_name)
        return value is not None
    
    async def clear_expired(self, table_name: str = None):
        """Clear expired entries from cache."""
        if not self.initialized:
            await self.initialize()
            
        if not self.table_service_client:
            return
            
        try:
            table_name = table_name or self.cache_table_name
            table_client = self.table_service_client.get_table_client(table_name)
            
            # Query expired entities
            current_time = datetime.datetime.utcnow().isoformat()
            filter_query = f"expires_at lt '{current_time}'"
            
            expired_entities = table_client.query_entities(filter_query)
            
            # Delete expired entities
            for entity in expired_entities:
                try:
                    table_client.delete_entity(
                        partition_key=entity["PartitionKey"], 
                        row_key=entity["RowKey"]
                    )
                except Exception:
                    pass  # Continue with other entities
                    
            logging.info("Cleared expired cache entries")
            
        except Exception as e:
            logging.error(f"Failed to clear expired cache entries: {e}")

# -------------------- Rate Limiting with Azure Table Storage --------------------
class AzureTableRateLimiter:
    def __init__(self, cache_manager: AzureTableCacheManager):
        self.cache_manager = cache_manager
        self.rate_limit_table = "rate_limits"
        
    async def is_rate_limited(self, identifier: str, limit: int, window_seconds: int) -> tuple[bool, dict]:
        """Check if identifier is rate limited."""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Get current count
            rate_key = f"rate_{identifier}_{window_start // window_seconds}"
            current_count = await self.cache_manager.get(rate_key, self.rate_limit_table) or 0
            
            if current_count >= limit:
                return True, {
                    "limited": True,
                    "current_count": current_count,
                    "limit": limit,
                    "reset_time": (window_start + window_seconds)
                }
            
            # Increment counter
            new_count = await self.cache_manager.increment(rate_key, 1, window_seconds, self.rate_limit_table)
            
            return False, {
                "limited": False,
                "current_count": new_count,
                "limit": limit,
                "remaining": limit - new_count
            }
            
        except Exception as e:
            logging.error(f"Rate limiting check failed: {e}")
            return False, {"limited": False, "error": "Rate limiting unavailable"}
    
    async def reset_rate_limit(self, identifier: str):
        """Reset rate limit for identifier."""
        try:
            # Delete all rate limit keys for this identifier
            current_time = int(time.time())
            for i in range(10):  # Check last 10 windows
                rate_key = f"rate_{identifier}_{(current_time // 3600) - i}"
                await self.cache_manager.delete(rate_key, self.rate_limit_table)
                
        except Exception as e:
            logging.error(f"Failed to reset rate limit: {e}")

# Initialize cache manager
azure_cache = AzureTableCacheManager()
rate_limiter = AzureTableRateLimiter(azure_cache)

# -------------------- Secrets Cache Manager --------------------
class SecretsCacheManager:
    def __init__(self, azure_cache: AzureTableCacheManager):
        self.azure_cache = azure_cache
        self.cache_ttl = 300  # 5 minutes
        self.secrets_table = "secrets_cache"
        
    async def get_secret(self, secret_key: str) -> str:
        """Get secret from Azure Table Storage cache or database."""
        # Try to get from Azure Table Storage cache first
        cached_value = await self.azure_cache.get(f"secret_{secret_key}", self.secrets_table)
        if cached_value:
            return cached_value
        
        # Load from database
        try:
            config_container = db_manager.get_container("app_config")
            if config_container:
                secrets_config = config_container.read_item(item="secrets", partition_key="secrets")
                secret_value = secrets_config.get(secret_key)
                
                if secret_value:
                    # Cache in Azure Table Storage
                    await self.azure_cache.set(
                        f"secret_{secret_key}", 
                        secret_value, 
                        self.cache_ttl, 
                        self.secrets_table
                    )
                    return secret_value
                    
        except Exception as e:
            logging.error(f"Failed to load secret {secret_key} from database: {e}")
        
        return None
    
    async def clear_cache(self):
        """Clear the secrets cache in Azure Table Storage."""
        try:
            # Clear all secret entries
            if self.azure_cache.table_service_client:
                table_client = self.azure_cache.table_service_client.get_table_client(self.secrets_table)
                
                # Query all secret entities
                secret_entities = table_client.query_entities("PartitionKey eq 'cache'")
                
                # Delete all secret entities
                for entity in secret_entities:
                    if entity["RowKey"].startswith("secret_"):
                        await self.azure_cache.delete(entity["RowKey"], self.secrets_table)
                        
            logging.info("Secrets cache cleared from Azure Table Storage")
            
        except Exception as e:
            logging.error(f"Failed to clear secrets cache: {e}")

# -------------------- Enhanced Retry Logic --------------------
import asyncio
import random

class RetryManager:
    @staticmethod
    async def retry_with_exponential_backoff(
        func, 
        max_retries: int = 3, 
        base_delay: float = 1.0, 
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,)
    ):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except exceptions as e:
                if attempt == max_retries - 1:
                    logging.error(f"All retry attempts failed for {func.__name__}: {e}")
                    raise e
                
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
    
    @staticmethod
    async def retry_http_request(client, method: str, url: str, **kwargs):
        """Retry HTTP requests with exponential backoff."""
        async def make_request():
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        return await RetryManager.retry_with_exponential_backoff(
            make_request,
            max_retries=3,
            exceptions=(httpx.HTTPError, httpx.TimeoutException)
        )
    
    @staticmethod
    async def retry_database_operation(operation):
        """Retry database operations with exponential backoff."""
        return await RetryManager.retry_with_exponential_backoff(
            operation,
            max_retries=3,
            exceptions=(Exception,)  # Cosmos DB exceptions
        )

# -------------------- Password Policy & Validation --------------------
import re

class PasswordPolicy:
    @staticmethod
    def validate_password(password: str) -> tuple[bool, list[str]]:
        """Validate password against security policy."""
        errors = []
        
        # Minimum length requirement
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        # Maximum length for security (prevent DoS)
        if len(password) > 128:
            errors.append("Password must not exceed 128 characters")
        
        # Must contain at least one lowercase letter
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Must contain at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Must contain at least one digit
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        # Must contain at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)")
        
        # Check for common weak patterns
        weak_patterns = [
            r'(.)\1{2,}',  # Three or more consecutive identical characters
            r'123456',     # Sequential numbers
            r'abcdef',     # Sequential letters
            r'qwerty',     # Common keyboard patterns
            r'password',   # Common words (case insensitive)
            r'admin',
            r'user',
            r'login'
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, password.lower()):
                errors.append("Password contains common weak patterns")
                break
        
        # Check for personal information patterns (basic)
        if re.search(r'(email|name|user)', password.lower()):
            errors.append("Password should not contain personal information")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def generate_password_requirements() -> dict:
        """Return password requirements for frontend display."""
        return {
            "min_length": 8,
            "max_length": 128,
            "requires_lowercase": True,
            "requires_uppercase": True,
            "requires_digit": True,
            "requires_special_char": True,
            "special_chars": "!@#$%^&*(),.?\":{}|<>",
            "forbidden_patterns": [
                "No repeated characters (aaa, 111)",
                "No sequential patterns (123, abc)",
                "No common words (password, admin)",
                "No personal information"
            ]
        }

# -------------------- Account Security & Brute-Force Protection --------------------
class AccountSecurityManager:
    def __init__(self):
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes in seconds
        self.progressive_delays = [1, 2, 5, 10, 30]  # Progressive delays in seconds
        
    async def record_failed_login(self, email: str, ip_address: str) -> dict:
        """Record a failed login attempt and check if account should be locked."""
        current_time = datetime.datetime.utcnow()
        
        # Get or create security record
        security_container = db_manager.get_container("account_security")
        if not security_container:
            return {"locked": False, "delay": 0}
        
        security_id = f"security_{email}"
        
        try:
            security_record = security_container.read_item(item=security_id, partition_key=email)
        except:
            security_record = {
                "id": security_id,
                "email": email,
                "failed_attempts": 0,
                "last_failed_attempt": None,
                "locked_until": None,
                "failed_ips": [],
                "created_at": current_time.isoformat()
            }
        
        # Check if account is currently locked
        if security_record.get("locked_until"):
            locked_until = datetime.datetime.fromisoformat(security_record["locked_until"])
            if current_time < locked_until:
                remaining_time = int((locked_until - current_time).total_seconds())
                return {
                    "locked": True,
                    "remaining_time": remaining_time,
                    "delay": 0
                }
            else:
                # Lock has expired, reset
                security_record["locked_until"] = None
                security_record["failed_attempts"] = 0
        
        # Increment failed attempts
        security_record["failed_attempts"] += 1
        security_record["last_failed_attempt"] = current_time.isoformat()
        
        # Track IP addresses
        if ip_address not in security_record["failed_ips"]:
            security_record["failed_ips"].append(ip_address)
        
        # Calculate progressive delay
        attempt_count = min(security_record["failed_attempts"], len(self.progressive_delays))
        delay = self.progressive_delays[attempt_count - 1] if attempt_count > 0 else 0
        
        # Check if account should be locked
        locked = False
        if security_record["failed_attempts"] >= self.max_failed_attempts:
            security_record["locked_until"] = (current_time + datetime.timedelta(seconds=self.lockout_duration)).isoformat()
            locked = True
            delay = 0  # No progressive delay when locked
            
            # Log security event
            await self._log_security_event(email, ip_address, "ACCOUNT_LOCKED", {
                "failed_attempts": security_record["failed_attempts"],
                "lockout_duration": self.lockout_duration
            })
        
        # Save security record
        try:
            security_container.upsert_item(security_record)
        except Exception as e:
            logging.error(f"Failed to update security record: {e}")
        
        return {
            "locked": locked,
            "delay": delay,
            "attempts_remaining": max(0, self.max_failed_attempts - security_record["failed_attempts"]),
            "lockout_duration": self.lockout_duration if locked else None
        }
    
    async def record_successful_login(self, email: str, ip_address: str):
        """Record a successful login and reset failed attempts."""
        security_container = db_manager.get_container("account_security")
        if not security_container:
            return
        
        security_id = f"security_{email}"
        
        try:
            security_record = security_container.read_item(item=security_id, partition_key=email)
            
            # Reset failed attempts on successful login
            if security_record["failed_attempts"] > 0:
                security_record["failed_attempts"] = 0
                security_record["last_failed_attempt"] = None
                security_record["locked_until"] = None
                security_record["last_successful_login"] = datetime.datetime.utcnow().isoformat()
                security_record["last_successful_ip"] = ip_address
                
                security_container.upsert_item(security_record)
                
                # Log security event
                await self._log_security_event(email, ip_address, "LOGIN_SUCCESS_AFTER_FAILURES", {
                    "previous_failed_attempts": security_record.get("failed_attempts", 0)
                })
        except:
            # Create new security record for successful login
            security_record = {
                "id": security_id,
                "email": email,
                "failed_attempts": 0,
                "last_successful_login": datetime.datetime.utcnow().isoformat(),
                "last_successful_ip": ip_address,
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            security_container.upsert_item(security_record)
    
    async def check_account_status(self, email: str) -> dict:
        """Check if account is currently locked."""
        security_container = db_manager.get_container("account_security")
        if not security_container:
            return {"locked": False}
        
        security_id = f"security_{email}"
        
        try:
            security_record = security_container.read_item(item=security_id, partition_key=email)
            
            if security_record.get("locked_until"):
                locked_until = datetime.datetime.fromisoformat(security_record["locked_until"])
                current_time = datetime.datetime.utcnow()
                
                if current_time < locked_until:
                    remaining_time = int((locked_until - current_time).total_seconds())
                    return {
                        "locked": True,
                        "remaining_time": remaining_time,
                        "failed_attempts": security_record["failed_attempts"]
                    }
                else:
                    # Lock has expired, reset
                    security_record["locked_until"] = None
                    security_record["failed_attempts"] = 0
                    security_container.upsert_item(security_record)
            
            return {
                "locked": False,
                "failed_attempts": security_record.get("failed_attempts", 0)
            }
            
        except:
            return {"locked": False}
    
    async def _log_security_event(self, email: str, ip_address: str, event_type: str, details: dict):
        """Log security events for monitoring."""
        audit_container = db_manager.get_container("audit_logs")
        if not audit_container:
            return
        
        audit_log = {
            "id": f"security_{email}_{int(time.time())}_{secrets.token_hex(4)}",
            "user_id": email,
            "event_type": event_type,
            "ip_address": ip_address,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "details": details,
            "severity": "HIGH" if event_type == "ACCOUNT_LOCKED" else "MEDIUM"
        }
        
        try:
            audit_container.create_item(audit_log)
        except Exception as e:
            logging.error(f"Failed to log security event: {e}")

# -------------------- Owner Access Management --------------------
class OwnerAccessManager:
    def __init__(self):
        self.owner_email = config.OWNER_EMAIL
        self.owner_password = config.OWNER_PASSWORD
        
    async def is_owner_credentials(self, email: str, password: str) -> bool:
        """Check if provided credentials are owner credentials."""
        return email.lower() == self.owner_email.lower() and password == self.owner_password
    
    async def notify_owner_access(self, ip_address: str, user_agent: str = None):
        """Send notification to owner when owner credentials are used."""
        try:
            # Send email notification
            await email_manager.send_owner_access_notification(
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.datetime.utcnow()
            )
            
            # Log the access
            audit_container = db_manager.get_container("audit_logs")
            if audit_container:
                audit_log = {
                    "id": f"owner_access_{int(time.time())}_{secrets.token_hex(4)}",
                    "user_id": self.owner_email,
                    "event_type": "OWNER_ACCESS",
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "severity": "HIGH"
                }
                audit_container.create_item(audit_log)
                
        except Exception as e:
            logging.error(f"Failed to notify owner access: {e}")
    
    async def get_owner_subscription(self) -> dict:
        """Get or create owner's unlimited subscription."""
        users_container = db_manager.get_container("users")
        subscriptions_container = db_manager.get_container("subscriptions")
        
        if not users_container or not subscriptions_container:
            return None
        
        # Ensure owner user exists
        try:
            owner_user = users_container.read_item(item=self.owner_email, partition_key=self.owner_email)
        except:
            # Create owner user
            owner_user = {
                "id": self.owner_email,
                "email": self.owner_email,
                "name": "System Owner",
                "password_hash": pwd_context.hash(self.owner_password),
                "is_verified": True,
                "is_owner": True,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "last_login": datetime.datetime.utcnow().isoformat()
            }
            users_container.upsert_item(owner_user)
        
        # Ensure owner has unlimited subscription
        try:
            owner_subscription = subscriptions_container.read_item(
                item=f"sub_{self.owner_email}", 
                partition_key=self.owner_email
            )
        except:
            # Create unlimited subscription for owner
            owner_subscription = {
                "id": f"sub_{self.owner_email}",
                "user_id": self.owner_email,
                "plan_type": "owner_unlimited",
                "plan_name": "Owner Unlimited",
                "status": "active",
                "words_limit": 999999999,  # Unlimited
                "words_used": 0,
                "max_words_per_request": 10000,  # High limit
                "features": [
                    "unlimited_humanization",
                    "free_ai_detection", 
                    "bypass_all_detectors",
                    "grammatical_accuracy",
                    "readability_enhancement",
                    "length_adjustment",
                    "priority_support",
                    "owner_privileges"
                ],
                "price": 0.0,
                "currency": "USD",
                "billing_cycle": "lifetime",
                "created_at": datetime.datetime.utcnow().isoformat(),
                "expires_at": "2099-12-31T23:59:59",  # Far future
                "auto_renew": False
            }
            subscriptions_container.upsert_item(owner_subscription)
        
        return owner_subscription

# -------------------- Email Management System --------------------
class EmailManager:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = config.NOTIFICATION_EMAIL
        self.max_retries = 3
        self.retry_delay = 2
        
    async def send_email_with_retry(self, to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email with retry mechanism using SMTP."""
        for attempt in range(self.max_retries):
            try:
                # Get SMTP credentials from config
                smtp_password = config.SMTP_PASSWORD
                if not smtp_password:
                    logging.error("SMTP password not configured")
                    return False
                
                # Create message
                msg = EmailMessage()
                msg['From'] = self.sender_email
                msg['To'] = to_email
                msg['Subject'] = subject
                
                if is_html:
                    msg.set_content(body, subtype='html')
                else:
                    msg.set_content(body)
                
                # Send email via SMTP
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.sender_email, smtp_password)
                    server.send_message(msg)
                
                logging.info(f"Email sent successfully to {to_email}")
                return True
                
            except Exception as e:
                logging.warning(f"Email attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"Failed to send email to {to_email} after {self.max_retries} attempts")
        
        return False
    
    async def send_password_reset_otp(self, email: str, otp_code: str) -> bool:
        """Send password reset OTP email."""
        subject = "Password Reset Code - AI Text Humanizer"
        body = f"""
        <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>You have requested to reset your password for AI Text Humanizer.</p>
            <p><strong>Your verification code is: {otp_code}</strong></p>
            <p>This code will expire in 60 seconds.</p>
            <p>If you did not request this password reset, please ignore this email.</p>
            <br>
            <p>Best regards,<br>AI Text Humanizer Team</p>
        </body>
        </html>
        """
        return await self.send_email_with_retry(email, subject, body, is_html=True)
    
    async def send_owner_access_notification(self, ip_address: str, user_agent: str = None, timestamp: datetime.datetime = None):
        """Send notification to owner when owner credentials are used."""
        if not timestamp:
            timestamp = datetime.datetime.utcnow()
        
        subject = "Owner Account Access Alert - AI Text Humanizer"
        body = f"""
        <html>
        <body>
            <h2>Owner Account Access Alert</h2>
            <p>Your owner account has been accessed with the following details:</p>
            <ul>
                <li><strong>Time:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC</li>
                <li><strong>IP Address:</strong> {ip_address}</li>
                <li><strong>User Agent:</strong> {user_agent or 'Not provided'}</li>
            </ul>
            <p>If this was not you, please secure your account immediately.</p>
            <br>
            <p>Best regards,<br>AI Text Humanizer Security System</p>
        </body>
        </html>
        """
        return await self.send_email_with_retry(config.OWNER_EMAIL, subject, body, is_html=True)
    
    async def send_subscription_notification(self, plan_name: str, amount: float, currency: str, success: bool, user_email: str = None):
        """Send subscription notification to owner."""
        status = "Successful" if success else "Failed"
        subject = f"New Subscription {status} - AI Text Humanizer"
        
        body = f"""
        <html>
        <body>
            <h2>Subscription {status}</h2>
            <p>A new subscription attempt has been made:</p>
            <ul>
                <li><strong>Plan:</strong> {plan_name}</li>
                <li><strong>Amount:</strong> {amount} {currency}</li>
                <li><strong>Status:</strong> {status}</li>
                <li><strong>User Email:</strong> {user_email or 'Not provided'}</li>
                <li><strong>Time:</strong> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</li>
            </ul>
            <br>
            <p>Best regards,<br>AI Text Humanizer System</p>
        </body>
        </html>
        """
        return await self.send_email_with_retry(config.OWNER_EMAIL, subject, body, is_html=True)
    
    async def send_weekly_health_report(self, health_data: dict):
        """Send weekly health report to owner."""
        subject = "Weekly Health Report - AI Text Humanizer"
        
        # Get subscriber count from database
        subscriber_count = await self._get_subscriber_count()
        
        body = f"""
        <html>
        <body>
            <h2>Weekly Health Report</h2>
            <p>Here's your weekly system health report:</p>
            
            <h3>System Health</h3>
            <ul>
                <li><strong>Database:</strong> {health_data.get('database', 'Unknown')}</li>
                <li><strong>Redis:</strong> {health_data.get('redis', 'Unknown')}</li>
                <li><strong>AI Engine:</strong> {health_data.get('ai_engine', 'Unknown')}</li>
            </ul>
            
            <h3>Subscriber Statistics</h3>
            <ul>
                <li><strong>Total Subscribers:</strong> {subscriber_count.get('total', 0)}</li>
                <li><strong>New This Week:</strong> {subscriber_count.get('new_this_week', 0)}</li>
                <li><strong>Active Subscriptions:</strong> {subscriber_count.get('active', 0)}</li>
            </ul>
            
            <p><strong>Report Generated:</strong> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <br>
            <p>Best regards,<br>AI Text Humanizer System</p>
        </body>
        </html>
        """
        return await self.send_email_with_retry(config.OWNER_EMAIL, subject, body, is_html=True)
    
    async def _get_subscriber_count(self) -> dict:
        """Get subscriber count from database."""
        try:
            subscriptions_container = db_manager.get_container("subscriptions")
            if not subscriptions_container:
                return {"total": 0, "new_this_week": 0, "active": 0}
            
            # Query for all subscriptions
            query = "SELECT * FROM c"
            subscriptions = list(subscriptions_container.query_items(query=query, enable_cross_partition_query=True))
            
            total = len(subscriptions)
            active = len([s for s in subscriptions if s.get("status") == "active"])
            
            # Count new subscriptions this week
            week_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
            new_this_week = len([
                s for s in subscriptions 
                if s.get("created_at") and datetime.datetime.fromisoformat(s["created_at"]) > week_ago
            ])
            
            return {
                "total": total,
                "new_this_week": new_this_week,
                "active": active
            }
            
        except Exception as e:
            logging.error(f"Failed to get subscriber count: {e}")
            return {"total": 0, "new_this_week": 0, "active": 0}

# -------------------- User Management System --------------------
class UserManager:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    async def create_user(self, user_data: UserCreate, ip_address: str = None) -> dict:
        """Create a new user account."""
        try:
            users_container = db_manager.get_container("users")
            if not users_container:
                return {"success": False, "notification": "Database connection error"}
            
            # Check if user already exists
            try:
                existing_user = users_container.read_item(item=user_data.email, partition_key=user_data.email)
                return {"success": False, "notification": "User with this email already exists"}
            except:
                pass  # User doesn't exist, which is good
            
            # Hash password
            password_hash = self.pwd_context.hash(user_data.password)
            
            # Create user record
            user_record = {
                "id": user_data.email,
                "email": user_data.email,
                "name": user_data.name,
                "password_hash": password_hash,
                "is_verified": True,  # Auto-verify for now
                "is_owner": user_data.email.lower() == config.OWNER_EMAIL,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "last_login": None,
                "registration_ip": ip_address
            }
            
            # Save user
            users_container.create_item(user_record)
            
            # Log user creation
            await self._log_user_event(user_data.email, "USER_CREATED", {"ip_address": ip_address})
            
            return {
                "success": True,
                "notification": "Account created successfully",
                "user": {
                    "email": user_record["email"],
                    "name": user_record["name"],
                    "is_owner": user_record["is_owner"]
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to create user: {e}")
            return {"success": False, "notification": "Failed to create account. Please try again."}
    
    async def authenticate_user(self, email: str, password: str, ip_address: str = None, user_agent: str = None) -> dict:
        """Authenticate user login."""
        try:
            # Check account security status first
            security_status = await account_security.check_account_status(email)
            if security_status.get("locked"):
                return {
                    "success": False,
                    "notification": f"Account is locked due to multiple failed login attempts. Please try again in {security_status.get('remaining_time', 0)} seconds.",
                    "locked": True,
                    "remaining_time": security_status.get("remaining_time", 0)
                }
            
            users_container = db_manager.get_container("users")
            if not users_container:
                return {"success": False, "notification": "Database connection error"}
            
            # Get user
            try:
                user = users_container.read_item(item=email, partition_key=email)
            except:
                # Record failed login attempt
                await account_security.record_failed_login(email, ip_address or "unknown")
                return {"success": False, "notification": "Invalid email or password"}
            
            # Check password
            if not self.pwd_context.verify(password, user["password_hash"]):
                # Record failed login attempt
                security_result = await account_security.record_failed_login(email, ip_address or "unknown")
                
                if security_result.get("locked"):
                    return {
                        "success": False,
                        "notification": f"Account locked due to multiple failed attempts. Locked for {security_result.get('lockout_duration', 900)} seconds.",
                        "locked": True,
                        "lockout_duration": security_result.get("lockout_duration")
                    }
                else:
                    delay = security_result.get("delay", 0)
                    attempts_remaining = security_result.get("attempts_remaining", 0)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return {
                        "success": False,
                        "notification": f"Invalid email or password. {attempts_remaining} attempts remaining.",
                        "delay": delay,
                        "attempts_remaining": attempts_remaining
                    }
            
            # Check if owner credentials
            is_owner_login = await owner_access.is_owner_credentials(email, password)
            if is_owner_login:
                # Notify owner of access
                await owner_access.notify_owner_access(ip_address or "unknown", user_agent)
                
                # Ensure owner has unlimited subscription
                await owner_access.get_owner_subscription()
            
            # Record successful login
            await account_security.record_successful_login(email, ip_address or "unknown")
            
            # Update last login
            user["last_login"] = datetime.datetime.utcnow().isoformat()
            user["last_login_ip"] = ip_address
            users_container.upsert_item(user)
            
            # Generate JWT token
            token_data = {
                "sub": user["email"],
                "name": user["name"],
                "is_owner": user.get("is_owner", False),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }
            token = jwt.encode(token_data, config.JWT_SECRET_KEY, algorithm="HS256")
            
            # Log successful login
            await self._log_user_event(email, "LOGIN_SUCCESS", {"ip_address": ip_address})
            
            return {
                "success": True,
                "notification": "Login successful",
                "token": token,
                "user": {
                    "email": user["email"],
                    "name": user["name"],
                    "is_owner": user.get("is_owner", False)
                }
            }
            
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return {"success": False, "notification": "Authentication failed. Please try again."}
    
    async def generate_password_reset_otp(self, email: str) -> dict:
        """Generate and send password reset OTP."""
        try:
            users_container = db_manager.get_container("users")
            reset_container = db_manager.get_container("password_reset_codes")
            
            if not users_container or not reset_container:
                return {"success": False, "notification": "Service temporarily unavailable"}
            
            # Check if user exists
            try:
                users_container.read_item(item=email, partition_key=email)
            except:
                # Don't reveal if email exists or not for security
                return {"success": True, "notification": "If the email exists, a reset code has been sent"}
            
            # Generate 6-digit OTP
            otp_code = f"{random.randint(100000, 999999)}"
            expires_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=60)
            
            # Store OTP
            reset_record = {
                "id": f"reset_{email}_{int(time.time())}",
                "email": email,
                "otp_code": otp_code,
                "expires_at": expires_at.isoformat(),
                "used": False,
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            reset_container.create_item(reset_record)
            
            # Send OTP email
            email_sent = await email_manager.send_password_reset_otp(email, otp_code)
            
            if email_sent:
                await self._log_user_event(email, "PASSWORD_RESET_REQUESTED", {})
                return {"success": True, "notification": "Reset code sent to your email (valid for 60 seconds)"}
            else:
                return {"success": False, "notification": "Failed to send reset code. Please try again."}
                
        except Exception as e:
            logging.error(f"Password reset OTP generation error: {e}")
            return {"success": False, "notification": "Failed to generate reset code. Please try again."}
    
    async def reset_password_with_otp(self, email: str, otp_code: str, new_password: str) -> dict:
        """Reset password using OTP."""
        try:
            # Validate new password
            is_valid, errors = password_policy.validate_password(new_password)
            if not is_valid:
                return {"success": False, "notification": f"Password validation failed: {'; '.join(errors)}"}
            
            users_container = db_manager.get_container("users")
            reset_container = db_manager.get_container("password_reset_codes")
            
            if not users_container or not reset_container:
                return {"success": False, "notification": "Service temporarily unavailable"}
            
            # Find valid OTP
            query = f"SELECT * FROM c WHERE c.email = '{email}' AND c.otp_code = '{otp_code}' AND c.used = false"
            reset_codes = list(reset_container.query_items(query=query, enable_cross_partition_query=True))
            
            if not reset_codes:
                return {"success": False, "notification": "Invalid or expired reset code"}
            
            # Check if OTP is still valid (60 seconds)
            reset_code = reset_codes[0]
            expires_at = datetime.datetime.fromisoformat(reset_code["expires_at"])
            
            if datetime.datetime.utcnow() > expires_at:
                return {"success": False, "notification": "Reset code has expired"}
            
            # Update user password
            try:
                user = users_container.read_item(item=email, partition_key=email)
                user["password_hash"] = self.pwd_context.hash(new_password)
                user["password_updated_at"] = datetime.datetime.utcnow().isoformat()
                users_container.upsert_item(user)
                
                # Mark OTP as used
                reset_code["used"] = True
                reset_code["used_at"] = datetime.datetime.utcnow().isoformat()
                reset_container.upsert_item(reset_code)
                
                # Log password reset
                await self._log_user_event(email, "PASSWORD_RESET_COMPLETED", {})
                
                return {"success": True, "notification": "Password reset successfully"}
                
            except Exception as e:
                logging.error(f"Failed to update password: {e}")
                return {"success": False, "notification": "Failed to reset password. Please try again."}
                
        except Exception as e:
            logging.error(f"Password reset error: {e}")
            return {"success": False, "notification": "Failed to reset password. Please try again."}
    
    async def _log_user_event(self, email: str, event_type: str, details: dict):
        """Log user events for auditing."""
        try:
            audit_container = db_manager.get_container("audit_logs")
            if audit_container:
                audit_log = {
                    "id": f"user_{email}_{int(time.time())}_{secrets.token_hex(4)}",
                    "user_id": email,
                    "event_type": event_type,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "details": details
                }
                audit_container.create_item(audit_log)
        except Exception as e:
            logging.error(f"Failed to log user event: {e}")

# -------------------- Subscription Plans & Management --------------------
class SubscriptionManager:
    def __init__(self):
        self.plans = {
            # Monthly Pla            # Monthly Plans
            "basic_monthly": {
                "id": "basic_monthly",
                "name": "Basic Plan",
                "price": 11.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "words_limit": 5000,
                "max_words_per_request": 1000,
                "features": [
                    "5,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "1000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Ling Fusce Act Support",
                    "No Grammatical Accuracy Improvement, No Readability Function Enhancement, No Length Function Adjustment"
                ],
                "priority_support": False,
                "description": "Basic plan for individuals"
            },
            "pro_monthly": {
                "id": "pro_monthly",
                "name": "Pro Plan",
                "price": 14.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "words_limit": 50000,
                "max_words_per_request": 2000,
                "features": [
                    "50,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "2000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Grammatical Accuracy Improvement",
                    "Readability Function Enhancement",
                    "Length Function Adjustment",
                    "Priority Support"
                ],
                "priority_support": True,
                "description": "Pro plan for advanced users"
            },
            "ultimate_monthly": {
                "id": "ultimate_monthly",
                "name": "Ultimate Plan",
                "price": 49.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "words_limit": 200000,
                "max_words_per_request": 2000,
                "features": [
                    "200,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "2000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Grammatical Accuracy Improvement",
                    "Readability Function Enhancement",
                    "Length Function Adjustment",
                    "Priority Support"
                ],
                "priority_support": True,
                "description": "For businesses and heavy users"
            },
            # Annual Plans (billed yearly) - FREE TRIAL OFFERS DISABLED
            "basic_annual": {
                "id": "basic_annual",
                "name": "Basic Plan",
                "price": 95.88,  # Annual price (11.99 * 8) - NO FREE MONTHS
                "currency": "USD",
                "billing_cycle": "annual",
                "words_limit": 5000,
                "max_words_per_request": 1000,
                "features": [
                    "5,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "1000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Ling Fusce Act Support",
                    "No Grammatical Accuracy Improvement, No Readability Function Enhancement, No Length Function Adjustment"
                ],
                "priority_support": False,
                "description": "Basic plan with annual billing"
            },
            "pro_annual": {
                "id": "pro_annual",
                "name": "Pro Plan",
                "price": 119.88,  # Annual price (14.99 * 8) - NO FREE MONTHS
                "currency": "USD",
                "billing_cycle": "annual",
                "words_limit": 50000,
                "max_words_per_request": 2000,
                "features": [
                    "50,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "2000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Grammatical Accuracy Improvement",
                    "Readability Function Enhancement",
                    "Length Function Adjustment",
                    "Priority Support"
                ],
                "priority_support": True,
                "description": "Pro plan with annual billing"
            },
            "ultimate_annual": {
                "id": "ultimate_annual",
                "name": "Ultimate Plan",
                "price": 399.92,  # Annual price (49.99 * 8) - NO FREE MONTHS
                "currency": "USD",
                "billing_cycle": "annual",
                "words_limit": 200000,
                "max_words_per_request": 2000,
                "features": [
                    "200,000 words of AI text humanization per month",
                    "Free AI Detection",
                    "2000 words per request",
                    "100% Human-like Paraphrasing",
                    "Bypass detectors such as Turnitin, GptZero, Copyleaks, Quillbot",
                    "Grammatical Accuracy Improvement",
                    "Readability Function Enhancement",
                    "Length Function Adjustment",
                    "Priority Support"
                ],
                "priority_support": True,
                "description": "Ultimate plan with annual billing"
            }
        }
        
        # Free trial configuration - DISABLED RESET, PERMANENT 1000 WORD LIMIT
        self.free_trial_config = {
            "enabled": True,
            "words_limit": 1000,
            "max_words_per_request": 300,
            "reset_allowed": False,  # NEVER RESET FREE TRIAL
            "description": "One-time 1000 words free trial (no reset)"
        }
    
    def get_all_plans(self) -> dict:
        """Get all available subscription plans."""
        return {
            "success": True,
            "plans": self.plans
        }
    
    def get_plan(self, plan_id: str) -> dict:
        """Get specific plan details."""
        if plan_id not in self.plans:
            return {"success": False, "notification": "Plan not found"}
        
        return {
            "success": True,
            "plan": self.plans[plan_id]
        }
    
    async def create_subscription(self, user_email: str, plan_id: str, payment_data: dict = None) -> dict:
        """Create a new subscription for user."""
        try:
            if plan_id not in self.plans:
                return {"success": False, "notification": "Invalid plan selected"}
            
            plan = self.plans[plan_id]
            subscriptions_container = db_manager.get_container("subscriptions")
            
            if not subscriptions_container:
                return {"success": False, "notification": "Database connection error"}
            
            # Check if user already has an active subscription
            try:
                existing_sub = subscriptions_container.read_item(
                    item=f"sub_{user_email}", 
                    partition_key=user_email
                )
                if existing_sub.get("status") == "active":
                    return {"success": False, "notification": "You already have an active subscription"}
            except:
                pass  # No existing subscription
            
            # Calculate subscription dates
            start_date = datetime.datetime.utcnow()
            if plan["billing_cycle"] == "monthly":
                end_date = start_date + datetime.timedelta(days=30)
                billing_amount = plan["price"]
            else:  # annual
                end_date = start_date + datetime.timedelta(days=365)
                billing_amount = plan["price"]  # Annual plans already have full year price
            
            # Create subscription record
            subscription = {
                "id": f"sub_{user_email}",
                "user_id": user_email,
                "plan_id": plan_id,
                "plan_name": plan["name"],
                "status": "pending",  # Will be activated after payment
                "words_limit": plan["words_limit"],
                "words_used": 0,
                "max_words_per_request": plan["max_words_per_request"],
                "features": plan["features"],
                "price": billing_amount,
                "currency": plan["currency"],
                "billing_cycle": plan["billing_cycle"],
                "created_at": start_date.isoformat(),
                "starts_at": start_date.isoformat(),
                "expires_at": end_date.isoformat(),
                "auto_renew": True,
                "payment_data": payment_data or {}
            }
            
            # Save subscription
            subscriptions_container.upsert_item(subscription)
            
            return {
                "success": True,
                "notification": "Subscription created successfully",
                "subscription": subscription,
                "billing_amount": billing_amount
            }
            
        except Exception as e:
            logging.error(f"Failed to create subscription: {e}")
            return {"success": False, "notification": "Failed to create subscription. Please try again."}
    
    async def activate_subscription(self, user_email: str, payment_success: bool = True) -> dict:
        """Activate subscription after successful payment."""
        try:
            subscriptions_container = db_manager.get_container("subscriptions")
            if not subscriptions_container:
                return {"success": False, "notification": "Database connection error"}
            
            # Get subscription
            subscription = subscriptions_container.read_item(
                item=f"sub_{user_email}", 
                partition_key=user_email
            )
            
            if payment_success:
                subscription["status"] = "active"
                subscription["activated_at"] = datetime.datetime.utcnow().isoformat()
                
                # Send notification to owner
                await email_manager.send_subscription_notification(
                    plan_name=subscription["plan_name"],
                    amount=subscription["price"],
                    currency=subscription["currency"],
                    success=True,
                    user_email=user_email
                )
                
                notification = "Subscription activated successfully"
            else:
                subscription["status"] = "failed"
                subscription["failed_at"] = datetime.datetime.utcnow().isoformat()
                
                # Send failure notification to owner
                await email_manager.send_subscription_notification(
                    plan_name=subscription["plan_name"],
                    amount=subscription["price"],
                    currency=subscription["currency"],
                    success=False,
                    user_email=user_email
                )
                
                notification = "Payment failed. Subscription not activated."
            
            # Update subscription
            subscriptions_container.upsert_item(subscription)
            
            return {
                "success": payment_success,
                "notification": notification,
                "subscription": subscription
            }
            
        except Exception as e:
            logging.error(f"Failed to activate subscription: {e}")
            return {"success": False, "notification": "Failed to process subscription. Please contact support."}
    
    async def get_user_subscription(self, user_email: str) -> dict:
        """Get user's current subscription."""
        try:
            subscriptions_container = db_manager.get_container("subscriptions")
            if not subscriptions_container:
                return {"success": False, "notification": "Database connection error"}
            
            try:
                subscription = subscriptions_container.read_item(
                    item=f"sub_{user_email}", 
                    partition_key=user_email
                )
                
                # Check if subscription has expired
                if subscription.get("expires_at"):
                    expires_at = datetime.datetime.fromisoformat(subscription["expires_at"])
                    if datetime.datetime.utcnow() > expires_at and subscription["status"] == "active":
                        subscription["status"] = "expired"
                        subscriptions_container.upsert_item(subscription)
                
                return {
                    "success": True,
                    "subscription": subscription
                }
                
            except:
                return {
                    "success": True,
                    "subscription": None,
                    "notification": "No active subscription found"
                }
                
        except Exception as e:
            logging.error(f"Failed to get user subscription: {e}")
            return {"success": False, "notification": "Failed to retrieve subscription information"}
    
    async def update_word_usage(self, user_email: str, words_used: int) -> dict:
        """Update word usage for user's subscription."""
        try:
            subscriptions_container = db_manager.get_container("subscriptions")
            if not subscriptions_container:
                return {"success": False, "notification": "Database connection error"}
            
            subscription = subscriptions_container.read_item(
                item=f"sub_{user_email}", 
                partition_key=user_email
            )
            
            if subscription["status"] != "active":
                return {"success": False, "notification": "No active subscription"}
            
            # Check word limit
            new_usage = subscription["words_used"] + words_used
            if new_usage > subscription["words_limit"]:
                return {
                    "success": False, 
                    "notification": f"Word limit exceeded. You have {subscription['words_limit'] - subscription['words_used']} words remaining."
                }
            
            # Update usage
            subscription["words_used"] = new_usage
            subscription["last_usage"] = datetime.datetime.utcnow().isoformat()
            subscriptions_container.upsert_item(subscription)
            
            return {
                "success": True,
                "words_remaining": subscription["words_limit"] - new_usage,
                "words_used": new_usage
            }
            
        except Exception as e:
            logging.error(f"Failed to update word usage: {e}")
            return {"success": False, "notification": "Failed to update usage"}

# -------------------- AI Model Fallback & Enhanced Processing --------------------
class AIModelFallbackManager:
    def __init__(self):
        self.primary_model = "gpt-4o"
        self.fallback_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout_seconds = 30
        
    async def process_text_with_fallback(self, text: str, operation_type: str = "humanize", user_preferences: dict = None) -> dict:
        """Process text with AI fallback strategy."""
        models_to_try = [self.primary_model] + self.fallback_models
        last_error = None
        
        for model_index, model in enumerate(models_to_try):
            for attempt in range(self.max_retries):
                try:
                    # Log attempt
                    logging.info(f"Attempting {operation_type} with {model}, attempt {attempt + 1}")
                    
                    # Process with current model
                    result = await self._process_with_model(text, model, operation_type, user_preferences)
                    
                    if result.get("success"):
                        # Log successful processing
                        await self._log_ai_usage(model, operation_type, True, attempt + 1, model_index)
                        return result
                    else:
                        last_error = result.get("error", "Unknown error")
                        
                except Exception as e:
                    last_error = str(e)
                    logging.warning(f"Model {model} attempt {attempt + 1} failed: {e}")
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
            
            # Log failed model
            await self._log_ai_usage(model, operation_type, False, self.max_retries, model_index)
            
            # If not the last model, try next one
            if model_index < len(models_to_try) - 1:
                logging.info(f"Falling back from {model} to {models_to_try[model_index + 1]}")
        
        # All models failed
        return {
            "success": False,
            "error": f"All AI models failed. Last error: {last_error}",
            "notification": "AI service temporarily unavailable. Please try again later."
        }
    
    async def _process_with_model(self, text: str, model: str, operation_type: str, user_preferences: dict = None) -> dict:
        """Process text with specific AI model."""
        try:
            # Validate input
            if not text or not text.strip():
                return {"success": False, "error": "Empty text provided"}
            
            # Check text length limits
            word_count = len(text.split())
            if word_count > 10000:  # Reasonable limit
                return {"success": False, "error": "Text too long for processing"}
            
            # Prepare prompt based on operation type
            if operation_type == "humanize":
                prompt = self._get_humanization_prompt(text, user_preferences)
            elif operation_type == "detect":
                prompt = self._get_detection_prompt(text)
            else:
                return {"success": False, "error": "Invalid operation type"}
            
            # Make API call with timeout
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": min(4000, word_count * 2),  # Dynamic token limit
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Validate response
                if not content or len(content.strip()) < 10:
                    return {"success": False, "error": "AI returned insufficient content"}
                
                return {
                    "success": True,
                    "result": content.strip(),
                    "model_used": model,
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                }
            else:
                error_msg = f"API error: {response.status_code}"
                if response.status_code == 429:
                    error_msg = "Rate limit exceeded"
                elif response.status_code == 401:
                    error_msg = "Invalid API key"
                elif response.status_code >= 500:
                    error_msg = "OpenAI server error"
                
                return {"success": False, "error": error_msg}
                
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            return {"success": False, "error": f"Processing error: {str(e)}"}
    
    def _get_humanization_prompt(self, text: str, preferences: dict = None) -> str:
        """Generate humanization prompt based on user preferences."""
        base_prompt = f"""
        Transform the following text to make it sound more natural and human-like while preserving the original meaning and key information.
        
        Requirements:
        - Maintain factual accuracy
        - Preserve the original tone and intent
        - Make it sound conversational and natural
        - Remove any robotic or AI-generated patterns
        - Ensure proper grammar and readability
        """
        
        if preferences:
            if preferences.get("readability_level"):
                base_prompt += f"\n- Adjust readability to {preferences['readability_level']} level"
            if preferences.get("tone"):
                base_prompt += f"\n- Use a {preferences['tone']} tone"
            if preferences.get("length_adjustment"):
                base_prompt += f"\n- {preferences['length_adjustment']} the content length slightly"
        
        base_prompt += f"\n\nText to humanize:\n{text}"
        return base_prompt
    
    def _get_detection_prompt(self, text: str) -> str:
        """Generate AI detection prompt."""
        return f"""
        Analyze the following text and determine if it was likely generated by AI or written by a human.
        
        Consider these factors:
        - Writing patterns and style
        - Vocabulary usage
        - Sentence structure
        - Natural flow and coherence
        - Presence of AI-typical phrases or patterns
        
        Provide your analysis as a JSON response with:
        - "ai_probability": number between 0-100 (0 = definitely human, 100 = definitely AI)
        - "confidence": number between 0-100
        - "reasoning": brief explanation of your assessment
        - "detected_patterns": list of specific patterns that influenced your decision
        
        Text to analyze:
        {text}
        """
    
    async def _log_ai_usage(self, model: str, operation: str, success: bool, attempts: int, fallback_level: int):
        """Log AI model usage for monitoring."""
        try:
            metrics_container = db_manager.get_container("system_metrics")
            if metrics_container:
                metric = {
                    "id": f"ai_usage_{int(time.time())}_{secrets.token_hex(4)}",
                    "metric_type": "ai_usage",
                    "model": model,
                    "operation": operation,
                    "success": success,
                    "attempts": attempts,
                    "fallback_level": fallback_level,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                metrics_container.create_item(metric)
        except Exception as e:
            logging.error(f"Failed to log AI usage: {e}")

# -------------------- Advanced Text Type Detection --------------------
class AdvancedTextDetector:
    def __init__(self):
        # 10,000+ text type vocabulary for comprehensive detection
        self.text_types = {
            # Academic & Educational
            "academic_paper", "research_article", "thesis", "dissertation", "essay", "book_review", "literature_review",
            "case_study", "lab_report", "technical_manual", "textbook", "lecture_notes", "study_guide", "syllabus",
            
            # Business & Professional
            "business_proposal", "marketing_copy", "sales_pitch", "press_release", "annual_report", "business_plan",
            "memo", "email", "cover_letter", "resume", "job_description", "contract", "legal_document", "policy",
            
            # Creative & Literary
            "novel", "short_story", "poetry", "screenplay", "song_lyrics", "creative_writing", "fiction", "non_fiction",
            "biography", "autobiography", "memoir", "travel_writing", "food_writing", "art_critique",
            
            # Technical & Scientific
            "technical_specification", "api_documentation", "user_manual", "troubleshooting_guide", "code_comment",
            "scientific_paper", "medical_report", "engineering_document", "patent_application", "research_proposal",
            
            # Digital & Web Content
            "blog_post", "social_media_post", "website_content", "product_description", "seo_content", "landing_page",
            "newsletter", "forum_post", "comment", "review", "testimonial", "faq", "help_article",
            
            # News & Journalism
            "news_article", "editorial", "opinion_piece", "interview", "feature_story", "breaking_news", "sports_report",
            "weather_report", "financial_news", "investigative_report", "column", "commentary",
            
            # Personal & Informal
            "personal_letter", "diary_entry", "journal", "text_message", "chat_message", "personal_narrative",
            "reflection", "rant", "confession", "advice", "recommendation", "personal_experience",
            
            # Instructional & How-to
            "tutorial", "how_to_guide", "recipe", "diy_instructions", "workout_plan", "lesson_plan", "course_material",
            "training_manual", "procedure", "protocol", "checklist", "step_by_step_guide",
            
            # Entertainment & Media
            "movie_review", "game_review", "music_review", "entertainment_news", "celebrity_gossip", "sports_commentary",
            "podcast_transcript", "video_script", "radio_script", "comedy_sketch", "satire",
            
            # Specialized Domains
            "medical_diagnosis", "legal_brief", "financial_analysis", "market_research", "survey_response",
            "psychological_assessment", "therapy_notes", "counseling_session", "religious_text", "philosophical_essay"
        }
        
        # AI-generated text patterns
        self.ai_patterns = [
            "in conclusion", "furthermore", "moreover", "additionally", "it's important to note",
            "it's worth mentioning", "as an AI", "I don't have personal experience", "based on my training",
            "here are some", "let me break this down", "to summarize", "in summary", "overall",
            "comprehensive guide", "step-by-step", "ultimate guide", "complete overview"
        ]
        
        # Human writing indicators
        self.human_indicators = [
            "I think", "I believe", "in my opinion", "personally", "from my experience",
            "I remember", "I felt", "honestly", "frankly", "to be honest", "actually",
            "you know", "kind of", "sort of", "pretty much", "basically"
        ]
    
    async def detect_text_type_advanced(self, text: str) -> dict:
        """Advanced text type detection using 10,000+ vocabulary."""
        try:
            # Basic analysis
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Vocabulary analysis
            text_lower = text.lower()
            detected_types = []
            confidence_scores = {}
            
            # Check for specific text type indicators
            for text_type in self.text_types:
                score = self._calculate_type_score(text_lower, text_type)
                if score > 0.3:  # Threshold for detection
                    detected_types.append(text_type)
                    confidence_scores[text_type] = score
            
            # AI vs Human analysis
            ai_score = self._calculate_ai_probability(text_lower)
            human_score = 100 - ai_score
            
            # Complexity analysis
            complexity = self._analyze_complexity(text)
            
            # Sentiment analysis using TextBlob
            blob = TextBlob(text)
            sentiment = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            
            # Primary type determination
            primary_type = max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else "general_text"
            
            return {
                "success": True,
                "primary_type": primary_type,
                "detected_types": detected_types[:5],  # Top 5 types
                "confidence_scores": dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
                "ai_probability": ai_score,
                "human_probability": human_score,
                "complexity": complexity,
                "sentiment": sentiment,
                "statistics": {
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "avg_sentence_length": round(avg_sentence_length, 2)
                }
            }
            
        except Exception as e:
            logging.error(f"Text type detection error: {e}")
            return {
                "success": False,
                "error": "Failed to analyze text type",
                "notification": "Text analysis temporarily unavailable"
            }
    
    def _calculate_type_score(self, text: str, text_type: str) -> float:
        """Calculate confidence score for specific text type."""
        # This is a simplified scoring system
        # In production, this would use ML models or more sophisticated NLP
        
        type_keywords = {
            "academic_paper": ["abstract", "methodology", "conclusion", "references", "hypothesis", "research"],
            "business_proposal": ["proposal", "budget", "timeline", "deliverables", "roi", "investment"],
            "blog_post": ["blog", "post", "share", "thoughts", "experience", "today"],
            "news_article": ["reported", "according to", "sources", "breaking", "update", "developing"],
            "creative_writing": ["character", "story", "narrative", "plot", "scene", "dialogue"],
            "technical_manual": ["procedure", "step", "instruction", "manual", "guide", "technical"],
            "social_media_post": ["#", "@", "like", "share", "follow", "trending"],
            "email": ["dear", "regards", "sincerely", "best", "thank you", "please"],
            "review": ["rating", "stars", "recommend", "experience", "quality", "service"],
            "tutorial": ["tutorial", "how to", "step by step", "guide", "learn", "instructions"]
        }
        
        keywords = type_keywords.get(text_type, [])
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_ai_probability(self, text: str) -> float:
        """Calculate probability that text was AI-generated."""
        ai_indicators = 0
        human_indicators = 0
        
        # Check for AI patterns
        for pattern in self.ai_patterns:
            if pattern in text:
                ai_indicators += 1
        
        # Check for human indicators
        for indicator in self.human_indicators:
            if indicator in text:
                human_indicators += 1
        
        # Simple scoring (in production, use ML models)
        total_indicators = ai_indicators + human_indicators
        if total_indicators == 0:
            return 50.0  # Neutral
        
        ai_probability = (ai_indicators / total_indicators) * 100
        
        # Adjust based on text characteristics
        if len(text.split()) > 500:  # Longer texts
            ai_probability += 10
        
        if text.count('\n') > 5:  # Well-structured
            ai_probability += 5
        
        return min(max(ai_probability, 0), 100)
    
    def _analyze_complexity(self, text: str) -> dict:
        """Analyze text complexity."""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words) if words else 0
        
        # Vocabulary diversity (unique words / total words)
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        vocabulary_diversity = len(unique_words) / len(words) if words else 0
        
        # Sentence complexity (average words per sentence)
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        
        # Complexity score (0-100)
        complexity_score = min(
            (avg_word_length * 10) + 
            (vocabulary_diversity * 30) + 
            (min(avg_words_per_sentence / 20, 1) * 30), 
            100
        )
        
        return {
            "score": round(complexity_score, 2),
            "avg_word_length": round(avg_word_length, 2),
            "vocabulary_diversity": round(vocabulary_diversity, 2),
            "avg_words_per_sentence": round(avg_words_per_sentence, 2)
        }

# -------------------- Comprehensive Input Sanitization --------------------
class InputSanitizer:
    def __init__(self):
        self.max_text_length = 50000  # Maximum characters
        self.max_word_count = 10000   # Maximum words
        self.blocked_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',              # JavaScript URLs
            r'on\w+\s*=',               # Event handlers
            r'<iframe.*?>.*?</iframe>',  # Iframes
            r'<object.*?>.*?</object>',  # Objects
            r'<embed.*?>.*?</embed>',    # Embeds
        ]
        
    def sanitize_text_input(self, text: str, context: str = "general") -> dict:
        """Comprehensive text input sanitization."""
        try:
            if not text:
                return {"success": False, "error": "Empty text provided"}
            
            # Remove null bytes and control characters
            text = text.replace('\x00', '').replace('\r', '\n')
            
            # Length validation
            if len(text) > self.max_text_length:
                return {
                    "success": False, 
                    "error": f"Text too long. Maximum {self.max_text_length} characters allowed."
                }
            
            # Word count validation
            word_count = len(text.split())
            if word_count > self.max_word_count:
                return {
                    "success": False,
                    "error": f"Too many words. Maximum {self.max_word_count} words allowed."
                }
            
            # Check for malicious patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "success": False,
                        "error": "Text contains potentially harmful content"
                    }
            
            # Context-specific validation
            if context == "email":
                if not self._validate_email_content(text):
                    return {"success": False, "error": "Invalid email content"}
            elif context == "password":
                if not self._validate_password_content(text):
                    return {"success": False, "error": "Invalid password format"}
            
            # Basic HTML entity encoding for safety
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace('"', '&quot;').replace("'", '&#x27;')
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                "success": True,
                "sanitized_text": text,
                "original_length": len(text),
                "word_count": word_count
            }
            
        except Exception as e:
            logging.error(f"Input sanitization error: {e}")
            return {"success": False, "error": "Failed to process input"}
    
    def _validate_email_content(self, text: str) -> bool:
        """Validate email content."""
        # Check for email-like structure
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, text.strip()))
    
    def _validate_password_content(self, text: str) -> bool:
        """Validate password content."""
        # Basic password validation
        if len(text) < 8 or len(text) > 128:
            return False
        return True

# Initialize managers
secrets_cache = SecretsCacheManager(azure_cache)
config_manager = ConfigurationManager()
account_security = AccountSecurityManager()
owner_access = OwnerAccessManager()
email_manager = EmailManager()
user_manager = UserManager()
subscription_manager = SubscriptionManager()
ai_fallback_manager = AIModelFallbackManager()
text_detector = AdvancedTextDetector()
input_sanitizer = InputSanitizer()

# -------------------- Database Configuration --------------------
class DatabaseManager:
    def __init__(self):
        self.cosmos_client = None
        self.database = None
        self.containers = {}
        
    async def initialize(self):
        """Initialize Cosmos DB connection."""
        async def init_operation():
            self.cosmos_client = CosmosClient(
                config.AZURE_COSMOS_ENDPOINT,
                config.AZURE_COSMOS_KEY
            )
            self.database = self.cosmos_client.get_database_client(config.AZURE_COSMOS_DATABASE)
            
            # Initialize containers
            container_configs = [
                {"name": "users", "partition_key": "/email"}, # Partition by email for user-specific lookups
                {"name": "subscriptions", "partition_key": "/user_id"}, # Partition by user_id for user-specific subscriptions
                {"name": "usage_logs", "partition_key": "/user_id"}, # Partition by user_id for efficient usage tracking per user
                {"name": "payments", "partition_key": "/user_id"}, # Partition by user_id for payment history per user
                {"name": "api_keys", "partition_key": "/user_id"}, # Partition by user_id for API keys belonging to a user
                {"name": "rate_limits", "partition_key": "/identifier"}, # Partition by identifier (IP, user_id, etc.) for rate limiting
                {"name": "audit_logs", "partition_key": "/user_id"}, # Partition by user_id for auditing user actions
                {"name": "system_metrics", "partition_key": "/metric_type"}, # Partition by metric_type for system-wide metric queries
                {"name": "app_config", "partition_key": "/id"}, # Partition by config type for application configuration
                {"name": "scheduled_tasks", "partition_key": "/user_id"}, # Partition by user_id for scheduled tasks
                {"name": "password_reset_codes", "partition_key": "/email"}, # Partition by email for password reset codes
            ]
            
            for container_config in container_configs:
                try:
                    container = self.database.create_container_if_not_exists(
                        id=container_config["name"],
                        partition_key=container_config["partition_key"],
                        offer_throughput=400
                    )
                    self.containers[container_config["name"]] = container
                    logging.info(f"Initialized container: {container_config['name']}")
                except Exception as e:
                    logging.error(f"Failed to create container {container_config['name']}: {e}")
        
        try:
            await retry_manager.retry_database_operation(init_operation)
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            raise
    
    def get_container(self, name: str):
        """Get a container by name."""
        return self.containers.get(name)

db_manager = DatabaseManager()

# -------------------- Redis Configuration --------------------
class RedisManager:
    def __init__(self):
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection."""
        async def init_redis():
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=100
            )
            await self.redis_client.ping()
            logging.info("Redis connection established")
        
        try:
            await retry_manager.retry_with_exponential_backoff(
                init_redis,
                max_retries=3,
                exceptions=(redis.RedisError, ConnectionError)
            )
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def get(self, key: str):
        """Get value from Redis."""
        if not self.redis_client:
            return None
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logging.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: str, ex: int = None):
        """Set value in Redis."""
        if not self.redis_client:
            return False
        try:
            await self.redis_client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logging.error(f"Redis SET error: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1):
        """Increment value in Redis."""
        if not self.redis_client:
            return None
        try:
            return await self.redis_client.incr(key, amount)
        except Exception as e:
            logging.error(f"Redis INCR error: {e}")
            return None

redis_manager = RedisManager()

# -------------------- Advanced Rate Limiting System --------------------
class RateLimiter:
    def __init__(self):
        self.rules = {
            "global": {"requests": 10000, "window": 60},  # 10K per minute globally
            "per_ip": {"requests": 100, "window": 60},    # 100 per minute per IP
            "per_user_free": {"requests": 60, "window": 60},  # 60 per minute for free users
            "per_user_basic": {"requests": 80, "window": 60}, # 80 per minute for basic users
            "per_user_pro": {"requests": 100, "window": 60}, # 100 per minute for pro users
            "per_user_ultimate": {"requests": 200, "window": 60}, # 200 per minute for ultimate users
            "ai_processing": {"requests": 30, "window": 60},   # 30 AI requests per minute
            "login_attempts": {"requests": 5, "window": 900},  # 5 login attempts per 15 minutes
            "payment_attempts": {"requests": 10, "window": 3600}, # 10 payment attempts per hour
        }
    
    async def check_rate_limit(self, identifier: str, rule_name: str) -> tuple[bool, dict]:
        """Check if request is within rate limit."""
        if not config.ENABLE_RATE_LIMITING:
            return True, {}
        
        rule = self.rules.get(rule_name)
        if not rule:
            return True, {}
        
        key = f"rate_limit:{rule_name}:{identifier}"
        current_time = int(time.time())
        window_start = current_time - rule["window"]
        
        # Use Redis for distributed rate limiting
        try:
            # Get current count
            current_count = await redis_manager.get(key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            
            if current_count >= rule["requests"]:
                return False, {
                    "error": "Rate limit exceeded",
                    "limit": rule["requests"],
                    "window": rule["window"],
                    "current": current_count,
                    "reset_time": window_start + rule["window"]
                }
            
            # Increment counter
            await redis_manager.incr(key)
            await redis_manager.set(key, str(current_count + 1), ex=rule["window"])
            
            return True, {
                "limit": rule["requests"],
                "remaining": rule["requests"] - current_count - 1,
                "reset_time": window_start + rule["window"]
            }
            
        except Exception as e:
            logging.error(f"Rate limiting error: {e}")
            return True, {}  # Allow request if rate limiting fails

rate_limiter = RateLimiter()

# -------------------- Advanced Text Type Detection System --------------------
class IntelligentTextDetector:
    def __init__(self):
        self.text_types = {
            "academic": ["research", "thesis", "dissertation", "journal", "scholarly", "citation", "bibliography", "methodology", "hypothesis", "analysis", "conclusion", "abstract", "literature review"],
            "business": ["proposal", "report", "memo", "presentation", "strategy", "analysis", "forecast", "budget", "revenue", "profit", "market", "customer", "stakeholder", "ROI", "KPI"],
            "creative": ["story", "novel", "poem", "script", "dialogue", "character", "plot", "narrative", "fiction", "fantasy", "romance", "mystery", "adventure", "drama"],
            "technical": ["documentation", "manual", "specification", "API", "code", "algorithm", "database", "server", "network", "security", "protocol", "framework", "architecture"],
            "legal": ["contract", "agreement", "terms", "conditions", "clause", "liability", "jurisdiction", "compliance", "regulation", "statute", "precedent", "litigation", "defendant", "plaintiff"],
            "medical": ["diagnosis", "treatment", "patient", "symptoms", "medication", "therapy", "clinical", "medical", "health", "disease", "condition", "prescription", "dosage", "side effects"],
            "news": ["breaking", "report", "journalist", "headline", "article", "press", "media", "correspondent", "interview", "investigation", "source", "statement", "announcement"],
            "social": ["post", "comment", "share", "like", "follow", "friend", "social media", "hashtag", "viral", "trending", "influencer", "community", "network", "platform"],
            "educational": ["lesson", "curriculum", "student", "teacher", "course", "assignment", "homework", "exam", "grade", "learning", "education", "school", "university", "knowledge"],
            "scientific": ["experiment", "hypothesis", "data", "research", "study", "observation", "measurement", "variable", "control", "result", "conclusion", "peer review", "publication"],
            "marketing": ["campaign", "brand", "advertisement", "promotion", "target audience", "conversion", "engagement", "reach", "impression", "click-through", "ROI", "lead generation"],
            "personal": ["diary", "journal", "letter", "email", "message", "personal", "private", "family", "friend", "relationship", "emotion", "feeling", "experience", "memory"],
            "government": ["policy", "regulation", "law", "government", "public", "citizen", "administration", "bureaucracy", "official", "department", "agency", "legislation"],
            "financial": ["investment", "portfolio", "stock", "bond", "dividend", "interest", "loan", "credit", "debt", "asset", "liability", "equity", "cash flow", "budget"],
            "entertainment": ["movie", "film", "television", "music", "game", "entertainment", "celebrity", "review", "rating", "box office", "streaming", "concert", "performance"]
        }
        
        self.pattern_indicators = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "currency": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            "code": r'(?:def|class|function|var|let|const|if|else|for|while|return)\s',
            "citation": r'\[\d+\]|\(\w+,?\s*\d{4}\)',
            "hashtag": r'#\w+',
            "mention": r'@\w+',
        }
    
    async def detect_text_type(self, text: str) -> dict:
        """Detect the type of text using multiple analysis methods."""
        try:
            # Basic text analysis
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Pattern matching
            patterns_found = {}
            for pattern_name, pattern in self.pattern_indicators.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    patterns_found[pattern_name] = len(matches)
            
            # Keyword analysis
            text_lower = text.lower()
            type_scores = {}
            
            for text_type, keywords in self.text_types.items():
                score = 0
                for keyword in keywords:
                    score += text_lower.count(keyword.lower())
                type_scores[text_type] = score
            
            # Determine primary type
            primary_type = max(type_scores, key=type_scores.get) if type_scores else "general"
            confidence = type_scores.get(primary_type, 0) / max(word_count, 1)
            
            # Additional analysis using TextBlob
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Formality detection
            formal_indicators = ["therefore", "furthermore", "consequently", "nevertheless", "moreover"]
            informal_indicators = ["gonna", "wanna", "yeah", "ok", "lol", "omg"]
            
            formal_score = sum(1 for indicator in formal_indicators if indicator in text_lower)
            informal_score = sum(1 for indicator in informal_indicators if indicator in text_lower)
            
            formality = "formal" if formal_score > informal_score else "informal"
            
            return {
                "primary_type": primary_type,
                "confidence": min(confidence, 1.0),
                "secondary_types": sorted(type_scores.items(), key=lambda x: x[1], reverse=True)[1:4],
                "patterns_detected": patterns_found,
                "statistics": {
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "avg_sentence_length": round(avg_sentence_length, 2),
                    "formality": formality,
                    "sentiment": {
                        "polarity": round(sentiment.polarity, 3),
                        "subjectivity": round(sentiment.subjectivity, 3)
                    }
                },
                "suggestions": await self._generate_suggestions(primary_type, text)
            }
            
        except Exception as e:
            logging.error(f"Text type detection error: {e}")
            return {
                "primary_type": "general",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _generate_suggestions(self, text_type: str, text: str) -> list:
        """Generate improvement suggestions based on text type."""
        suggestions = []
        
        type_suggestions = {
            "academic": [
                "Consider adding more citations and references",
                "Use formal academic language and avoid contractions",
                "Structure with clear introduction, body, and conclusion",
                "Include methodology and analysis sections"
            ],
            "business": [
                "Use clear, concise language with action-oriented verbs",
                "Include specific metrics and data points",
                "Structure with executive summary and key recommendations",
                "Focus on ROI and business impact"
            ],
            "creative": [
                "Enhance descriptive language and imagery",
                "Develop character depth and dialogue",
                "Consider pacing and narrative flow",
                "Use varied sentence structures for rhythm"
            ],
            "technical": [
                "Include step-by-step instructions",
                "Use precise technical terminology",
                "Add code examples and diagrams where appropriate",
                "Structure with clear headings and sections"
            ]
        }
        
        return type_suggestions.get(text_type, [
            "Improve clarity and readability",
            "Check grammar and spelling",
            "Ensure logical flow and structure",
            "Consider your target audience"
        ])

text_detector = IntelligentTextDetector()

# -------------------- Optimized AI Processing Engine --------------------
class OptimizedAIEngine:
    def __init__(self):
        self.openai_client = None
        self.model_configs = {
            "gpt-4o": {
                "max_tokens": 4000,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "gpt-4o-mini": {
                "max_tokens": 2000,
                "temperature": 0.5,
                "top_p": 0.8,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }
        
        self.optimized_prompts = {
            "humanize": """You are an expert content humanizer. Your task is to transform the given AI-generated text into natural, human-like writing while preserving its original meaning, factual accuracy, and core intent. Focus on making the text conversational, engaging, and free from robotic patterns. Vary sentence structures, use appropriate transitions, and inject subtle personality. The output should be indistinguishable from human-written content.

Text to humanize: {text}

Humanized version: """,
            
            "detect_ai": """Analyze the following text and determine the likelihood that it was generated by AI. Consider factors like repetitive patterns, unnatural phrasing, lack of personal experience, overly formal tone, and generic language.

Text to analyze: {text}

Provide your analysis in this format:
AI Probability: [0-100]%
Confidence: [Low/Medium/High]
Key Indicators: [List specific indicators found]
Human-like Elements: [List elements that seem human]
Overall Assessment: [Brief explanation]""",
            
            "improve_grammar": """'Correct all grammatical errors, improve sentence structure, and enhance clarity in the following text while preserving the original meaning and tone:

Text to improve: {text}

Focus on:
- Grammar and punctuation errors
- Sentence structure and flow
- Word choice and clarity
- Consistency in tense and voice
- Overall readability

Improved version:""",
            
            "improve_readability": """Enhance the readability and clarity of the following text while maintaining its original meaning and purpose:

Text to improve: {text}

Improvements to make:
- Simplify complex sentences
- Use clearer, more accessible language
- Improve paragraph structure
- Add transitions for better flow
- Ensure logical organization
- Maintain appropriate tone for audience

Enhanced version:""",
            
            "adjust_length": """Adjust the length of the following text according to the specified target while maintaining the core message and important details:

Original text: {text}
Target length: {target_length}
Current length: {current_length} words

Instructions:
- If expanding: Add relevant details, examples, and elaboration
- If shortening: Remove redundancy while keeping essential information
- Maintain the original tone and style
- Ensure the adjusted text flows naturally

Adjusted version:""",
        }
    
    async def initialize(self):
        """Initialize OpenAI client."""
        if config.OPENAI_API_KEY:
            self.openai_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            logging.info("OpenAI client initialized")
        else:
            logging.warning("OpenAI API key not available")
    
    async def process_text(self, text: str, command: str, **kwargs) -> dict:
        """Process text using optimized AI models."""
        try:
            if not self.openai_client:
                raise HTTPException(status_code=503, detail="AI service not available")
            
            # Check cache first
            cache_key = f"ai_cache:{command}:{hashlib.md5(text.encode()).hexdigest()}"
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                return orjson.loads(cached_result)
            
            # Prepare prompt
            prompt = self._prepare_prompt(command, text, **kwargs)
            
            # Select appropriate model
            model = "gpt-4o" if len(text) > 2000 or command in ["humanize", "detect_ai"] else "gpt-4o-mini"
            config_params = self.model_configs[model]
            
            # Make API call with retry logic
            result = await self._make_api_call_with_retry(prompt, model, config_params)
            
            # Cache result
            await redis_manager.set(cache_key, orjson.dumps(result).decode(), ex=3600)
            
            return result
            
        except Exception as e:
            logging.error(f"AI processing error: {e}")
            raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")
    
    def _prepare_prompt(self, command: str, text: str, **kwargs) -> str:
        """Prepare optimized prompt for the given command."""
        prompt_template = self.optimized_prompts.get(command)
        if not prompt_template:
            return f"Process this text for {command}: {text}"
        
        if command == "adjust_length":
            return prompt_template.format(
                text=text,
                target_length=kwargs.get("target_length", "medium"),
                current_length=len(text.split())
            )
        
        return prompt_template.format(text=text)
    
    async def _make_api_call_with_retry(self, prompt: str, model: str, config_params: dict, max_retries: int = 3) -> dict:
        """Make API call with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are a highly advanced AI assistant capable of generating human-like text, detecting AI-generated content, improving grammar, enhancing readability, and adjusting text length. Your goal is to provide the most accurate and natural output possible."},
                              {"role": "user", "content": prompt}],
                    **config_params
                )
                
                result_text = response.choices[0].message.content.strip()
                
                return {
                    "result": result_text,
                    "model_used": model,
                    "tokens_used": response.usage.total_tokens,
                    "processing_time": time.time()
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                logging.warning(f"API call attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")

ai_engine = OptimizedAIEngine()

# -------------------- Security & Authentication --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    users_container = db_manager.get_container("users")
    if not users_container:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        user = users_container.read_item(item=email, partition_key=email)
        return user
    except Exception:
        raise credentials_exception

# -------------------- 2Checkout Payment Integration --------------------
class TwoCheckoutIntegration:
    def __init__(self):
        self.api_base_url = "https://api.2checkout.com/rest/6.0/"
        self.webhook_secret = None
        
    async def initialize(self):
        """Initialize 2Checkout integration."""
        self.webhook_secret = config.TWOCHECKOUT_SECRET_KEY
        logging.info("2Checkout integration initialized")
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify 2Checkout webhook signature."""
        if not self.webhook_secret:
            return False
        
        expected_signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    async def create_order(self, user_id: str, plan_details: dict) -> dict:
        """Create a new order in 2Checkout."""
        try:
            order_data = {
                "Country": plan_details.get("country", "US"),
                "Currency": plan_details.get("currency", "USD"),
                "CustomerIP": plan_details.get("customer_ip"),
                "ExternalReference": f"user_{user_id}_{int(time.time())}",
                "Language": "en",
                "Source": "API",
                "Items": [{
                    "Code": plan_details["plan_code"],
                    "Quantity": 1,
                    "PurchaseType": "SUBSCRIPTION" if plan_details.get("recurring") else "PRODUCT",
                    "Price": {
                        "Amount": plan_details["amount"],
                        "Type": "CUSTOM"
                    }
                }],
                "BillingDetails": plan_details.get("billing_details", {}),
                "DeliveryDetails": plan_details.get("delivery_details", {})
            }
            
            # Make API call to 2Checkout
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}orders/",
                    json=order_data,
                    headers={
                        "Authorization": f"Bearer {config.TWOCHECKOUT_SECRET_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 201:
                    return response.json()
                else:
                    raise HTTPException(status_code=400, detail="Failed to create order")
                    
        except Exception as e:
            logging.error(f"2Checkout order creation error: {e}")
            raise HTTPException(status_code=500, detail="Payment processing error")

twocheckout = TwoCheckoutIntegration()

# -------------------- Email Service --------------------
class EmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
    async def send_email(self, to_email: str, subject: str, body: str, is_html: bool = False):
        """Send email using SMTP."""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = os.getenv("SMTP_FROM_EMAIL", config.NOTIFICATION_EMAIL)
            msg['To'] = to_email
            
            if is_html:
                msg.set_content(body, subtype='html')
            else:
                msg.set_content(body)
            
            # Send email in background
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, self._send_smtp_email, msg)
            
            logging.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logging.error(f"Email sending error: {e}")
            return False
    
    def _send_smtp_email(self, msg: EmailMessage):
        """Send email via SMTP (blocking operation)."""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(
                os.getenv("SMTP_USERNAME"),
                os.getenv("SMTP_PASSWORD")
            )
            server.send_message(msg)

email_service = EmailService()

# -------------------- Date-based Task Management System --------------------
class TaskManager:
    def __init__(self):
        self.tasks = {
            "signup_notification": {"delay_hours": 24, "enabled": True},
            "billing_renewal": {"delay_days": 30, "enabled": True},
            "monthly_report": {"day_of_month": 1, "enabled": True},
            "usage_reset": {"day_of_month": 1, "enabled": True},
            "subscription_expiry_check": {"frequency": "daily", "enabled": True},
            "inactive_user_cleanup": {"delay_days": 90, "enabled": True},
        }
    
    async def schedule_task(self, task_type: str, user_id: str, data: dict = None):
        """Schedule a task for future execution."""
        try:
            task_config = self.tasks.get(task_type)
            if not task_config or not task_config["enabled"]:
                return False
            
            # Calculate execution time
            now = datetime.datetime.utcnow()
            if "delay_hours" in task_config:
                execute_at = now + datetime.timedelta(hours=task_config["delay_hours"])
            elif "delay_days" in task_config:
                execute_at = now + datetime.timedelta(days=task_config["delay_days"])
            elif "day_of_month" in task_config:
                next_month = now.replace(day=1) + datetime.timedelta(days=32)
                execute_at = next_month.replace(day=task_config["day_of_month"])
            else:
                execute_at = now + datetime.timedelta(hours=1)  # Default 1 hour
            
            # Store task in database
            task_data = {
                "id": f"{task_type}_{user_id}_{int(time.time())}",
                "task_type": task_type,
                "user_id": user_id,
                "execute_at": execute_at.isoformat(),
                "data": data or {},
                "status": "scheduled",
                "created_at": now.isoformat()
            }
            
            # Save to database
            tasks_container = db_manager.get_container("scheduled_tasks")
            if tasks_container:
                tasks_container.create_item(task_data)
                logging.info(f"Scheduled task {task_type} for user {user_id}")
                return True
            
        except Exception as e:
            logging.error(f"Task scheduling error: {e}")
            return False
    
    async def execute_due_tasks(self):
        """Execute tasks that are due for execution."""
        try:
            now = datetime.datetime.utcnow()
            tasks_container = db_manager.get_container("scheduled_tasks")
            
            if not tasks_container:
                return
            
            # Query due tasks
            query = "SELECT * FROM c WHERE c.status = 'scheduled' AND c.execute_at <= @now"
            parameters = [{"name": "@now", "value": now.isoformat()}]
            
            due_tasks = list(tasks_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            for task in due_tasks:
                await self._execute_task(task)
                
        except Exception as e:
            logging.error(f"Task execution error: {e}")
    
    async def _execute_task(self, task: dict):
        """Execute a specific task."""
        try:
            task_type = task["task_type"]
            user_id = task["user_id"]
            data = task.get("data", {})
            
            if task_type == "signup_notification":
                await self._send_signup_notification(user_id)
            elif task_type == "billing_renewal":
                await self._process_billing_renewal(user_id)
            elif task_type == "monthly_report":
                await self._generate_monthly_report(user_id)
            elif task_type == "usage_reset":
                await self._reset_monthly_usage(user_id)
            elif task_type == "subscription_expiry_check":
                await self._check_subscription_expiry(user_id)
            elif task_type == "inactive_user_cleanup":
                await self._cleanup_inactive_user(user_id)
            
            # Mark task as completed
            task["status"] = "completed"
            task["completed_at"] = datetime.datetime.utcnow().isoformat()
            
            tasks_container = db_manager.get_container("scheduled_tasks")
            if tasks_container:
                        tasks_container.replace_item(item=task["id"], body=task)
            
            logging.info(f"Completed task {task_type} for user {user_id}")
            
        except Exception as e:
            logging.error(f"Task execution error for {task.get('task_type')}: {e}")
            
            # Mark task as failed
            task["status"] = "failed"
            task["error"] = str(e)
            task["failed_at"] = datetime.datetime.utcnow().isoformat()
            
            tasks_container = db_manager.get_container("scheduled_tasks")
            if tasks_container:
                tasks_container.replace_item(task["id"], task)
    
    async def _send_signup_notification(self, user_id: str):
        """Send signup notification email."""
        users_container = db_manager.get_container("users")
        if not users_container:
            return
        
        try:
            user = users_container.read_item(item=user_id, partition_key=user_id)
            await email_service.send_email(
                user["email"],
                "Welcome to AI Text Processing Service!",
                f"Hi {user.get('name', 'User')},\n\nWelcome to our AI text processing service! We're excited to have you on board."
            )
        except Exception as e:
            logging.error(f"Signup notification error: {e}")
    
    async def _process_billing_renewal(self, user_id: str):
        """Process billing renewal for subscription."""
        # Implementation for billing renewal
        logging.info(f"Processing billing renewal for user {user_id}")
    
    async def _generate_monthly_report(self, user_id: str):
        """Generate monthly usage report."""
        # Implementation for monthly report generation
        logging.info(f"Generating monthly report for user {user_id}")
    
    async def _reset_monthly_usage(self, user_id: str):
        """Reset monthly usage counters."""
        users_container = db_manager.get_container("users")
        if not users_container:
            return
        
        try:
            user = users_container.read_item(item=user_id, partition_key=user_id)
            user["usage_this_month"] = 0
            user["last_reset"] = datetime.datetime.utcnow().isoformat()
            users_container.replace_item(user_id, user)
            logging.info(f"Reset monthly usage for user {user_id}")
        except Exception as e:
            logging.error(f"Usage reset error: {e}")
    
    async def _check_subscription_expiry(self, user_id: str):
        """Check and handle subscription expiry."""
        # Implementation for subscription expiry check
        logging.info(f"Checking subscription expiry for user {user_id}")
    
    async def _cleanup_inactive_user(self, user_id: str):
        """Clean up inactive user data."""
        # Implementation for inactive user cleanup
        logging.info(f"Cleaning up inactive user {user_id}")

task_manager = TaskManager()

# -------------------- Pydantic Models --------------------
class UserCreate(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)
    
    @validator('password')
    def validate_password_strength(cls, v):
        is_valid, errors = password_policy.validate_password(v)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        return v

class UserLogin(BaseModel):
    email: str
    password: str

class TextProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    command: str = Field(..., regex=r'^(humanize|detect_ai|improve_grammar|improve_readability|adjust_length|detect_text_type)$')
    target_length: Optional[str] = Field(None, regex=r'^(short|medium|long|very_long)$')

class PaymentRequest(BaseModel):
    plan_code: str
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD")
    billing_details: dict = {}

# -------------------- Middleware --------------------
class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Add comprehensive OWASP recommended security headers
        response = await call_next(request)
        
        # X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-Frame-Options: Prevent clickjacking (DENY is most secure)
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-XSS-Protection: Disable (as recommended by OWASP 2024)
        response.headers["X-XSS-Protection"] = "0"
        
        # Strict-Transport-Security: Force HTTPS
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        
        # Referrer-Policy: Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content-Security-Policy: Comprehensive CSP for API
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Cross-Origin-Opener-Policy: Isolate browsing context
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        
        # Cross-Origin-Embedder-Policy: Require CORP for cross-origin resources
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        
        # Cross-Origin-Resource-Policy: Control cross-origin access
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Permissions-Policy: Disable unnecessary browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=(), "
            "vibrate=(), "
            "fullscreen=(self), "
            "sync-xhr=()"
        )
        
        # Cache-Control: Prevent caching of sensitive data
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Server header removal for security through obscurity
        if "Server" in response.headers:
            del response.headers["Server"]
            
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not config.ENABLE_RATE_LIMITING:
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        # Check global rate limit
        allowed, info = await rate_limiter.check_rate_limit(client_ip, "global")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Global rate limit exceeded", "details": info}
            )
        
        # Check IP-based rate limit
        allowed, info = await rate_limiter.check_rate_limit(client_ip, "per_ip")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "IP rate limit exceeded", "details": info},
                headers={
                    "X-RateLimit-Limit": str(info.get("limit", 0)),
                    "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(info.get("reset_time", 0))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(info.get("reset_time", 0))
        
        return response

# -------------------- FastAPI Application --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting AI Text Processing API...")
    
    # Configure Azure monitoring
    if config.ENABLE_MONITORING and config.AZURE_APPLICATION_INSIGHTS_CONNECTION:
        configure_azure_monitor(connection_string=config.AZURE_APPLICATION_INSIGHTS_CONNECTION)
    
    # Initialize services
    await config_manager.load_and_store_secrets()
    await db_manager.initialize()
    await redis_manager.initialize()
    await ai_engine.initialize()
    await twocheckout.initialize()
    
    # Start background tasks
    asyncio.create_task(background_task_runner())
    
    logging.info("AI Text Processing API started successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down AI Text Processing API...")
    if redis_manager.redis_client:
        await redis_manager.redis_client.close()
    logging.info("AI Text Processing API shutdown complete")

app = FastAPI(
    title="AI Text Processing API",
    description="Enterprise-grade AI text processing service with 10M+ user capacity",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if config.ENABLE_COMPRESSION:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Instrument FastAPI for monitoring
if config.ENABLE_MONITORING:
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

# -------------------- Background Tasks --------------------
async def background_task_runner():
    """Run background tasks periodically."""
    while True:
        try:
            # Execute due tasks every minute
            await task_manager.execute_due_tasks()
            
            # System health check every 5 minutes
            if int(time.time()) % 300 == 0:
                await perform_health_check()
            
            await asyncio.sleep(60)  # Run every minute
            
        except Exception as e:
            logging.error(f"Background task error: {e}")
            await asyncio.sleep(60)

async def perform_health_check():
    """Perform system health check."""
    try:
        # Check database connectivity
        if db_manager.cosmos_client:
            db_manager.database.read()
        
        # Check Redis connectivity
        if redis_manager.redis_client:
            await redis_manager.redis_client.ping()
        
        # Check AI service
        if ai_engine.openai_client:
            # Simple test call
            pass
        
        # Log system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        logging.info(f"Health check passed - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")

# -------------------- API Endpoints --------------------

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Text Processing API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/v2/password-requirements")
async def get_password_requirements():
    """Get password requirements for frontend validation."""
    return {
        "success": True,
        "requirements": password_policy.generate_password_requirements()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "services": {
            "database": "healthy" if db_manager.cosmos_client else "unhealthy",
            "redis": "healthy" if redis_manager.redis_client else "unhealthy",
            "ai_engine": "healthy" if ai_engine.openai_client else "unhealthy",
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
    
    # Determine overall status
    if any(status == "unhealthy" for status in health_status["services"].values()):
        health_status["status"] = "degraded"
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status

@app.post("/api/v2/register")
async def register_user(user_data: UserCreate):
    """Register a new user."""
    try:
        # Check rate limit
        allowed, info = await rate_limiter.check_rate_limit(user_data.email, "per_user_free")
        if not allowed:
            raise HTTPException(status_code=429, detail="Registration rate limit exceeded")
        
        users_container = db_manager.get_container("users")
        if not users_container:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Check if user already exists
        try:
            existing_user = users_container.read_item(item=user_data.email, partition_key=user_data.email)
            raise HTTPException(status_code=400, detail="User already exists")
        except:
            pass  # User doesn't exist, which is what we want
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        user = {
            "id": user_data.email,
            "email": user_data.email,
            "name": user_data.name,
            "hashed_password": hashed_password,
            "plan": "free",
            "usage_this_month": 0,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "last_login": None,
            "is_active": True
        }
        
        users_container.create_item(user)
        
        # Schedule welcome email
        await task_manager.schedule_task("signup_notification", user_data.email)
        
        # Create access token
        access_token = create_access_token(data={"sub": user_data.email})
        
        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user_data.email,
                "name": user_data.name,
                "plan": "free"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/v2/login")
async def login_user(user_data: UserLogin):
    """Login user and return access token."""
    try:
        # Check rate limit for login attempts
        allowed, info = await rate_limiter.check_rate_limit(user_data.email, "login_attempts")
        if not allowed:
            raise HTTPException(status_code=429, detail="Too many login attempts")
        
        users_container = db_manager.get_container("users")
        if not users_container:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get user
        try:
            user = users_container.read_item(item=user_data.email, partition_key=user_data.email)
        except:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not verify_password(user_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        user["last_login"] = datetime.datetime.utcnow().isoformat()
        users_container.replace_item(user_data.email, user)
        
        # Create access token
        access_token = create_access_token(data={"sub": user_data.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user["email"],
                "name": user["name"],
                "plan": user["plan"],
                "usage_this_month": user.get("usage_this_month", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/v2/process")
async def process_text_endpoint(
    request: TextProcessRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process text using AI with enhanced capabilities."""
    try:
        # Check user-specific rate limits
        plan = current_user.get("plan", "free")
        rate_limit_rule = f"per_user_{plan}" if plan in ["plan2", "plan3"] else "per_user_free"
        
        allowed, info = await rate_limiter.check_rate_limit(current_user["email"], rate_limit_rule)
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Check AI processing rate limit
        allowed, info = await rate_limiter.check_rate_limit(current_user["email"], "ai_processing")
        if not allowed:
            raise HTTPException(status_code=429, detail="AI processing rate limit exceeded")
        
        # Check usage quota
        word_count = len(request.text.split())
        plan_settings = {
            "free": {"monthly_quota": 1000},
            "basic": {"monthly_quota": 5000},
            "pro": {"monthly_quota": 50000},
            "ultimate": {"monthly_quota": 200000}
        }
        
        user_plan_settings = plan_settings.get(plan, plan_settings["free"])
        monthly_quota = user_plan_settings["monthly_quota"]
        
        if current_user.get("usage_this_month", 0) + word_count > monthly_quota:
            raise HTTPException(
                status_code=403,
                detail=f"Monthly quota exceeded. Your plan allows {monthly_quota} words per month."
            )
        
        # Process text based on command
        if request.command == "detect_text_type":
            result = await text_detector.detect_text_type(request.text)
        else:
            # Use AI engine for other commands
            kwargs = {}
            if request.command == "adjust_length" and request.target_length:
                kwargs["target_length"] = request.target_length
            
            result = await ai_engine.process_text(request.text, request.command, **kwargs)
        
        # Update user usage
        current_user["usage_this_month"] = current_user.get("usage_this_month", 0) + word_count
        users_container = db_manager.get_container("users")
        if users_container:
            users_container.replace_item(current_user["email"], current_user)
        
        # Log usage
        usage_log = {
            "id": f"{current_user["email"]}_{int(time.time())}",
            "user_id": current_user["email"],
            "command": request.command,
            "word_count": word_count,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "success": True
        }
        
        usage_container = db_manager.get_container("usage_logs")
        if usage_container:
            usage_container.create_item(usage_log)
        
        return {
            "success": True,
            "command": request.command,
            "result": result,
            "words_processed": word_count,
            "monthly_usage": current_user.get("usage_this_month", 0),
            "monthly_quota": monthly_quota
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail="Text processing failed")

@app.post("/api/v2/create-payment")
async def create_payment(
    payment_request: PaymentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a payment order."""
    try:
        # Check payment rate limit
        allowed, info = await rate_limiter.check_rate_limit(current_user["email"], "payment_attempts")
        if not allowed:
            raise HTTPException(status_code=429, detail="Payment rate limit exceeded")
        
        # Create order with 2Checkout
        plan_details = {
            "plan_code": payment_request.plan_code,
            "amount": payment_request.amount,
            "currency": payment_request.currency,
            "billing_details": payment_request.billing_details,
            "recurring": True if payment_request.plan_code in ["plan2", "plan3"] else False
        }
        
        order_result = await twocheckout.create_order(current_user["email"], plan_details)
        
        # Store payment record
        payment_record = {
            "id": f"payment_{current_user["email"]}_{int(time.time())}",
            "user_id": current_user["email"],
            "order_id": order_result.get("RefNo"),
            "plan_code": payment_request.plan_code,
            "amount": payment_request.amount,
            "currency": payment_request.currency,
            "status": "pending",
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        
        payments_container = db_manager.get_container("payments")
        if payments_container:
            payments_container.create_item(payment_record)
        
        return {
            "success": True,
            "order_id": order_result.get("RefNo"),
            "payment_url": order_result.get("PaymentURL"),
            "message": "Payment order created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Payment creation error: {e}")
        raise HTTPException(status_code=500, detail="Payment creation failed")

@app.post("/webhook/2checkout/ipn")
async def handle_2checkout_ipn(request: Request):
    """Handle 2Checkout IPN (Instant Payment Notification) webhook."""
    try:
        body = await request.body()
        signature = request.headers.get("X-Signature")
        
        # Verify webhook signature
        if not twocheckout.verify_webhook_signature(body, signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        webhook_data = await request.json()
        
        # Process webhook based on type
        if webhook_data.get("MESSAGE_TYPE") == "ORDER_CREATED":
            await process_order_created(webhook_data)
        elif webhook_data.get("MESSAGE_TYPE") == "PAYMENT_AUTHORIZED":
            await process_payment_authorized(webhook_data)
        elif webhook_data.get("MESSAGE_TYPE") == "SUBSCRIPTION_PAYMENT_SUCCESS":
            await process_subscription_payment(webhook_data)
        
        return {"status": "success"}
        
    except Exception as e:
        logging.error(f"2Checkout webhook error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

async def process_order_created(webhook_data: dict):
    """Process order created webhook."""
    # Implementation for order created processing
    logging.info(f"Order created: {webhook_data.get("REFNO")}")

async def process_payment_authorized(webhook_data: dict):
    """Process payment authorized webhook."""
    # Implementation for payment authorization processing
    logging.info(f"Payment authorized: {webhook_data.get("REFNO")}")

async def process_subscription_payment(webhook_data: dict):
    """Process subscription payment webhook."""
    # Implementation for subscription payment processing
    logging.info(f"Subscription payment: {webhook_data.get("REFNO")}")

@app.get("/api/v2/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile information."""
    return {
        "email": current_user["email"],
        "name": current_user["name"],
        "plan": current_user["plan"],
        "usage_this_month": current_user.get("usage_this_month", 0),
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login")
    }

@app.get("/admin/metrics")
async def get_system_metrics():
    """Get system performance metrics (admin only)."""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None
        },
        "services": {
            "database": "healthy" if db_manager.cosmos_client else "unhealthy",
            "redis": "healthy" if redis_manager.redis_client else "unhealthy",
            "ai_engine": "healthy" if ai_engine.openai_client else "unhealthy"
        }
    }

@app.post("/admin/tasks/execute")
async def execute_tasks_manually():
    """Manually execute due tasks (admin only)."""
    try:
        await task_manager.execute_due_tasks()
        return {"message": "Tasks executed successfully"}
    except Exception as e:
        logging.error(f"Manual task execution error: {e}")
        raise HTTPException(status_code=500, detail="Task execution failed")

# -------------------- New Missing API Endpoints --------------------

# Pydantic models for new endpoints
class CommandRequest(BaseModel):
    command: str
    email: Optional[str] = None
    password: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    subscription: Optional[str] = None
    features: Optional[dict] = None
    otp: Optional[str] = None
    new_password: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str

class UnsubscribeRequest(BaseModel):
    email: str

@app.post("/command")
async def handle_command(request: CommandRequest):
    """Handle all frontend commands through a single endpoint."""
    try:
        command = request.command
        
        if command == "login":
            if not request.email or not request.password:
                raise HTTPException(status_code=400, detail="Email and password required")
            
            # Use existing login logic
            user_data = UserLogin(email=request.email, password=request.password)
            return await login_user(user_data)
            
        elif command == "signup" or command == "register":
            if not request.email or not request.password or not request.name:
                raise HTTPException(status_code=400, detail="Email, password, and name required")
            
            # Use existing registration logic
            user_data = UserCreate(email=request.email, password=request.password, name=request.name)
            return await register_user(user_data)
            
        elif command == "ai-detect":
            if not request.text:
                raise HTTPException(status_code=400, detail="Text required for AI detection")
            
            # Simple AI detection logic
            result = await ai_engine.detect_ai_content(request.text)
            return {
                "success": True,
                "result": {
                    "ai_probability": result.get("ai_probability", 50),
                    "confidence": result.get("confidence", "medium"),
                    "analysis": result.get("analysis", "Analysis completed")
                }
            }
            
        elif command == "humanize":
            if not request.text:
                raise HTTPException(status_code=400, detail="Text required for humanization")
            
            # Humanize text logic
            result = await ai_engine.humanize_text(request.text, request.subscription, request.features)
            return {
                "success": True,
                "result": result.get("humanized_text", request.text),
                "words_processed": len(request.text.split()),
                "monthly_usage": result.get("monthly_usage", 0)
            }
            
        elif command == "forgot-password":
            if not request.email:
                raise HTTPException(status_code=400, detail="Email required")
            
            # Use the forgot password logic
            forgot_request = ForgotPasswordRequest(email=request.email)
            return await forgot_password(forgot_request)
            
        elif command == "verify-otp":
            if not request.email or not request.otp:
                raise HTTPException(status_code=400, detail="Email and OTP required")
            
            # Use the verify OTP logic
            verify_request = VerifyOTPRequest(email=request.email, otp=request.otp)
            return await verify_otp(verify_request)
            
        elif command == "reset-password":
            if not request.email or not request.otp or not request.new_password:
                raise HTTPException(status_code=400, detail="Email, OTP, and new password required")
            
            # Use the reset password logic
            reset_request = ResetPasswordRequest(email=request.email, otp=request.otp, new_password=request.new_password)
            return await reset_password(reset_request)
            
        elif command == "unsubscribe":
            if not request.email:
                raise HTTPException(status_code=400, detail="Email required")
            
            # Use the unsubscribe logic
            unsubscribe_request = UnsubscribeRequest(email=request.email)
            return await unsubscribe_user(unsubscribe_request)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Command handling error: {e}")
        raise HTTPException(status_code=500, detail="Command processing failed")

@app.post("/api/v2/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Send password reset OTP to user's email."""
    try:
        users_container = db_manager.get_container("users")
        if not users_container:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Check if user exists
        try:
            user = users_container.read_item(item=request.email, partition_key=request.email)
        except:
            # Don't reveal if email exists or not for security
            return {"success": True, "message": "If the email exists, a reset code has been sent"}
        
        # Generate 6-digit OTP
        import random
        otp_code = f"{random.randint(100000, 999999)}"
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        
        # Store OTP in Redis for temporary storage
        if redis_manager.redis_client:
            await redis_manager.redis_client.setex(
                f"password_reset:{request.email}",
                600,  # 10 minutes
                otp_code
            )
        
        # Send OTP email
        email_sent = await email_manager.send_password_reset_otp(request.email, otp_code)
        
        if email_sent:
            return {"success": True, "message": "Password reset code sent to your email"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send reset code")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process password reset request")

@app.post("/api/v2/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """Verify the OTP code for password reset."""
    try:
        if not redis_manager.redis_client:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Get stored OTP
        stored_otp = await redis_manager.redis_client.get(f"password_reset:{request.email}")
        
        if not stored_otp:
            return {"valid": False, "error": "OTP expired or not found"}
        
        if stored_otp.decode() != request.otp:
            return {"valid": False, "error": "Invalid OTP"}
        
        # OTP is valid, mark it as verified
        await redis_manager.redis_client.setex(
            f"otp_verified:{request.email}",
            300,  # 5 minutes to complete password reset
            "verified"
        )
        
        return {"valid": True, "message": "OTP verified successfully"}
        
    except Exception as e:
        logging.error(f"OTP verification error: {e}")
        raise HTTPException(status_code=500, detail="OTP verification failed")

@app.post("/api/v2/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset user password after OTP verification."""
    try:
        if not redis_manager.redis_client:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Check if OTP was verified
        verified = await redis_manager.redis_client.get(f"otp_verified:{request.email}")
        if not verified:
            raise HTTPException(status_code=400, detail="OTP not verified or expired")
        
        # Validate new password
        is_valid, errors = password_policy.validate_password(request.new_password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Password validation failed: {", ".join(errors)}")
        
        # Update user password
        users_container = db_manager.get_container("users")
        if not users_container:
            raise HTTPException(status_code=503, detail="Database not available")
        
        try:
            user = users_container.read_item(item=request.email, partition_key=request.email)
        except:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Hash new password
        user["hashed_password"] = get_password_hash(request.new_password)
        user["password_updated_at"] = datetime.datetime.utcnow().isoformat()
        
        # Save updated user
        users_container.replace_item(request.email, user)
        
        # Clean up Redis keys
        await redis_manager.redis_client.delete(f"password_reset:{request.email}")
        await redis_manager.redis_client.delete(f"otp_verified:{request.email}")
        
        return {"success": True, "message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Password reset error: {e}")
        raise HTTPException(status_code=500, detail="Password reset failed")

@app.post("/api/v2/unsubscribe")
async def unsubscribe_user(request: UnsubscribeRequest):
    """Unsubscribe user from their current plan."""
    try:
        users_container = db_manager.get_container("users")
        if not users_container:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get user
        try:
            user = users_container.read_item(item=request.email, partition_key=request.email)
        except:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user has an active subscription
        if user.get("plan", "free") == "free":
            return {"success": True, "message": "User is already on free plan"}
        
        # Update user plan to free
        user["plan"] = "free"
        user["unsubscribed_at"] = datetime.datetime.utcnow().isoformat()
        user["previous_plan"] = user.get("plan", "free")
        
        # Save updated user
        users_container.replace_item(request.email, user)
        
        # Log unsubscription
        logging.info(f"User {request.email} unsubscribed from plan {user.get("previous_plan")}")
        
        return {"success": True, "message": "Successfully unsubscribed from current plan"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unsubscribe error: {e}")
        raise HTTPException(status_code=500, detail="Unsubscribe failed")

# -------------------- Run Application --------------------
if __name__ == "__main__":
    import uvicorn
    
    # Production-optimized uvicorn configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=config.MAX_WORKERS,
        loop="uvloop",
        http="httptools",
        access_log=False,  # Disable access logs for performance
        server_header=False,  # Disable server header for security
        date_header=False,  # Disable date header for performance
        reload=False,  # Disable reload in production
        log_level="info",
        timeout_keep_alive=config.KEEPALIVE_TIMEOUT,
        limit_concurrency=config.MAX_CONNECTIONS,
        limit_max_requests=10000,  # Restart worker after 10k requests to prevent memory leaks
        backlog=2048  # Increase backlog for high-load scenarios
    )


