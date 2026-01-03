"""Earn service for Process Factory - integrates with Bybit Earn API."""

from typing import Any

from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.models import BybitEarnHolding, BybitEarnProduct
from hean.logging import get_logger

logger = get_logger(__name__)


class EarnService:
    """Service for managing Earn products via Bybit API."""

    def __init__(self, http_client: BybitHTTPClient) -> None:
        """Initialize Earn service.

        Args:
            http_client: Bybit HTTP client instance
        """
        self._http_client = http_client

    async def list_products(
        self,
        category: str | None = None,
        coin: str | None = None,
    ) -> list[BybitEarnProduct]:
        """List available Earn products.

        Args:
            category: Optional category filter (e.g., "FlexibleSaving", "FixedSaving", "OnChain")
            coin: Optional coin filter (e.g., "USDT", "BTC")

        Returns:
            List of Earn products
        """
        try:
            products_data = await self._http_client.get_earn_products(
                category=category, coin=coin
            )
            products = []
            for product_data in products_data:
                try:
                    # Parse product data (adjust based on actual API response format)
                    product = BybitEarnProduct(
                        product_id=str(product_data.get("productId", "")),
                        category=str(product_data.get("category", "")),
                        coin=str(product_data.get("coin", "")),
                        product_name=str(product_data.get("productName", "")),
                        purchase_currency=str(product_data.get("purchaseCurrency", "")),
                        interest_rate=(
                            float(product_data["interestRate"])
                            if product_data.get("interestRate")
                            else None
                        ),
                        min_purchase_amount=product_data.get("minPurchaseAmount"),
                        max_purchase_amount=product_data.get("maxPurchaseAmount"),
                        redemption_delay_days=(
                            int(product_data["redemptionDelayDays"])
                            if product_data.get("redemptionDelayDays") is not None
                            else None
                        ),
                        lock_period_days=(
                            int(product_data["lockPeriodDays"])
                            if product_data.get("lockPeriodDays") is not None
                            else None
                        ),
                        auto_compound=bool(product_data.get("autoCompound", False)),
                        extra=product_data,
                    )
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to parse product data: {e}, skipping")
                    continue
            return products
        except Exception as e:
            logger.error(f"Failed to list Earn products: {e}")
            raise

    async def get_holdings(
        self,
        category: str | None = None,
        coin: str | None = None,
        product_id: str | None = None,
    ) -> list[BybitEarnHolding]:
        """Get current Earn holdings.

        Args:
            category: Optional category filter
            coin: Optional coin filter
            product_id: Optional product ID filter

        Returns:
            List of current holdings
        """
        try:
            holdings_data = await self._http_client.get_earn_holdings(
                category=category, coin=coin, product_id=product_id
            )
            holdings = []
            for holding_data in holdings_data:
                try:
                    holding = BybitEarnHolding(
                        product_id=str(holding_data.get("productId", "")),
                        category=str(holding_data.get("category", "")),
                        coin=str(holding_data.get("coin", "")),
                        quantity=str(holding_data.get("quantity", "0")),
                        total_interest=holding_data.get("totalInterest"),
                        next_interest_payment_time=holding_data.get("nextInterestPaymentTime"),
                        next_redemption_time=holding_data.get("nextRedemptionTime"),
                        auto_compound=bool(holding_data.get("autoCompound", False)),
                        extra=holding_data,
                    )
                    holdings.append(holding)
                except Exception as e:
                    logger.warning(f"Failed to parse holding data: {e}, skipping")
                    continue
            return holdings
        except Exception as e:
            logger.error(f"Failed to get Earn holdings: {e}")
            raise

    async def stake(
        self,
        category: str,
        account_type: str,
        amount: str,
        coin: str,
        product_id: str,
    ) -> dict[str, Any]:
        """Stake (invest) in an Earn product.

        Args:
            category: Product category (e.g., "FlexibleSaving", "OnChain")
            account_type: Account type ("FUND" or "UNIFIED")
            amount: Amount to stake (as string)
            coin: Coin symbol (e.g., "USDT", "BTC")
            product_id: Product ID

        Returns:
            Order response dict
        """
        try:
            response = await self._http_client.place_earn_order(
                category=category,
                order_type="Stake",
                account_type=account_type,
                amount=amount,
                coin=coin,
                product_id=product_id,
            )
            logger.info(
                f"Earn stake order placed: {amount} {coin} in product {product_id} "
                f"(category: {category})"
            )
            return response
        except Exception as e:
            logger.error(f"Failed to place Earn stake order: {e}")
            raise

    async def redeem(
        self,
        category: str,
        account_type: str,
        amount: str,
        coin: str,
        product_id: str,
    ) -> dict[str, Any]:
        """Redeem (withdraw) from an Earn product.

        Args:
            category: Product category (e.g., "FlexibleSaving", "OnChain")
            account_type: Account type ("FUND" or "UNIFIED")
            amount: Amount to redeem (as string, or "ALL" for full redemption)
            coin: Coin symbol (e.g., "USDT", "BTC")
            product_id: Product ID

        Returns:
            Order response dict
        """
        try:
            response = await self._http_client.place_earn_order(
                category=category,
                order_type="Redeem",
                account_type=account_type,
                amount=amount,
                coin=coin,
                product_id=product_id,
            )
            logger.info(
                f"Earn redeem order placed: {amount} {coin} from product {product_id} "
                f"(category: {category})"
            )
            return response
        except Exception as e:
            logger.error(f"Failed to place Earn redeem order: {e}")
            raise

