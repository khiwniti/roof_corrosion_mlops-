"""Tile fetch module — retrieves satellite tiles from Maxar/Nearmap APIs.

This module handles:
1. Geocoding addresses → lat/lng
2. Fetching tiles from Maxar SecureWatch or Nearmap MapBrowser API
3. Caching tiles in S3 to minimize API costs
4. Tile stitching for large AOIs
"""

import hashlib
import io
from pathlib import Path
from typing import Optional

import boto3
import httpx
import rasterio
from pydantic import BaseModel
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling


class TileRequest(BaseModel):
    lat: float
    lng: float
    zoom: int = 20  # ~0.3m/pixel at zoom 20 for Maxar
    source: str = "maxar"  # maxar | nearmap
    size_px: int = 512


class TileFetcher:
    """Fetch satellite tiles from commercial APIs with S3 caching."""

    MAXAR_TILE_URL = (
        "https://services.maxar.com/earthservice/v1/tile"
        "?z={zoom}&x={x}&y={y}&coverage=VIVID2"
    )
    NEARMAP_TILE_URL = (
        "https://us0.nearmap.com/maps"
        "?x={x}&y={y}&z={zoom}&nml=Vert&http=true"
    )

    def __init__(
        self,
        maxar_api_key: Optional[str] = None,
        nearmap_api_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "tiles/",
    ):
        self.maxar_api_key = maxar_api_key
        self.nearmap_api_key = nearmap_api_key
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3") if s3_bucket else None

    def _tile_coords(self, lat: float, lng: float, zoom: int) -> tuple[int, int]:
        """Convert lat/lng to tile coordinates (Slippy map convention)."""
        import math

        n = 2**zoom
        x = int((lng + 180) / 360 * n)
        lat_rad = math.radians(lat)
        y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
        return x, y

    def _cache_key(self, source: str, zoom: int, x: int, y: int) -> str:
        """Generate S3 cache key for a tile."""
        return f"{self.s3_prefix}{source}/{zoom}/{x}/{y}.tif"

    async def fetch_tile(self, request: TileRequest) -> bytes:
        """Fetch a single tile, checking S3 cache first.

        Returns:
            Raw tile bytes (GeoTIFF or PNG)
        """
        x, y = self._tile_coords(request.lat, request.lng, request.zoom)
        cache_key = self._cache_key(request.source, request.zoom, x, y)

        # Check S3 cache
        if self.s3_client and self.s3_bucket:
            try:
                obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=cache_key)
                return obj["Body"].read()
            except self.s3_client.exceptions.NoSuchKey:
                pass  # Cache miss, fetch from API

        # Fetch from tile API
        if request.source == "maxar":
            tile_bytes = await self._fetch_maxar(request.zoom, x, y)
        elif request.source == "nearmap":
            tile_bytes = await self._fetch_nearmap(request.zoom, x, y)
        else:
            raise ValueError(f"Unknown tile source: {request.source}")

        # Cache in S3
        if self.s3_client and self.s3_bucket:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=cache_key,
                Body=tile_bytes,
                ContentType="image/tiff" if tile_bytes.startswith(b"II") else "image/png",
            )

        return tile_bytes

    async def _fetch_maxar(self, zoom: int, x: int, y: int) -> bytes:
        """Fetch tile from Maxar SecureWatch API."""
        url = self.MAXAR_TILE_URL.format(zoom=zoom, x=x, y=y)
        headers = {}
        if self.maxar_api_key:
            headers["Authorization"] = f"Bearer {self.maxar_api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content

    async def _fetch_nearmap(self, zoom: int, x: int, y: int) -> bytes:
        """Fetch tile from Nearmap MapBrowser API."""
        url = self.NEARMAP_TILE_URL.format(zoom=zoom, x=x, y=y)
        params = {}
        if self.nearmap_api_key:
            params["apikey"] = self.nearmap_api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.content

    async def fetch_aoi(
        self,
        lat: float,
        lng: float,
        radius_m: float = 100,
        zoom: int = 20,
        source: str = "maxar",
    ) -> Path:
        """Fetch a stitched mosaic for an area of interest.

        Args:
            lat, lng: center coordinates
            radius_m: radius in meters around center
            zoom: tile zoom level
            source: tile provider

        Returns:
            Path to stitched GeoTIFF
        """
        # Calculate tile range
        import math

        center_x, center_y = self._tile_coords(lat, lng, zoom)
        # Approximate: at zoom 20, each tile covers ~150m
        tile_span_m = 40075016.686 / (2**zoom) * 256 / 256  # rough
        tiles_needed = max(1, int(radius_m / tile_span_m) + 1)

        # Fetch all tiles in range
        tiles = []
        for dx in range(-tiles_needed, tiles_needed + 1):
            for dy in range(-tiles_needed, tiles_needed + 1):
                req = TileRequest(
                    lat=lat, lng=lng, zoom=zoom, source=source,
                )
                # Override x/y directly
                x = center_x + dx
                y = center_y + dy
                try:
                    tile_bytes = await self.fetch_tile(req)
                    tiles.append((x, y, tile_bytes))
                except Exception:
                    continue  # skip missing tiles at edges

        if not tiles:
            raise RuntimeError(f"No tiles fetched for AOI at ({lat}, {lng})")

        # Stitch tiles into a single GeoTIFF
        # TODO: implement mosaic stitching with rasterio.merge
        # For now, return the center tile
        center_tile = tiles[len(tiles) // 2] if tiles else None
        if center_tile:
            output_path = Path(f"/tmp/aoi_{hashlib.md5(f'{lat},{lng}'.encode()).hexdigest()}.tif")
            output_path.write_bytes(center_tile[2])
            return output_path

        raise RuntimeError("Failed to fetch any tiles for AOI")
