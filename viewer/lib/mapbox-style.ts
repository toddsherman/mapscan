type StyleLayerLike = {
  id: string;
  type?: string;
  "source-layer"?: string;
};

/**
 * Return an opaque-water insertion point for thematic rasters.
 *
 * `waterway` is commonly linework, so its presence says nothing about whether
 * it can hide low-zoom raster pixels that straddle the coast. Prefer the
 * canonical water fill, then a fill backed by the `water` source layer, and
 * finally another explicitly water-named fill. Returning undefined is safer
 * than pretending linework is a mask.
 */
export function findWaterFillInsertionLayer(
  layers: readonly StyleLayerLike[] | undefined,
) {
  if (!layers) return undefined;

  const canonical = layers.find((layer) => layer.id === "water" && layer.type === "fill");
  if (canonical) return canonical.id;

  const sourceWater = layers.find(
    (layer) =>
      layer.type === "fill" &&
      layer["source-layer"] === "water" &&
      !/(?:shadow|outline|label)/i.test(layer.id),
  );
  if (sourceWater) return sourceWater.id;

  return layers.find(
    (layer) =>
      layer.type === "fill" &&
      /(^|[-_])water($|[-_])/i.test(layer.id) &&
      !/(?:shadow|outline|label)/i.test(layer.id),
  )?.id;
}

/**
 * Return the first layer above the basemap's complete fill stack.
 *
 * Thematic rasters belong above opaque land-use fills, otherwise developed
 * areas look like missing data. Linework and symbols should remain above the
 * thematic surface, so insert immediately after the final fill rather than at
 * the top of the style.
 */
export function findThematicRasterInsertionLayer(
  layers: readonly StyleLayerLike[] | undefined,
) {
  if (!layers?.length) return undefined;
  let finalFillIndex = -1;
  layers.forEach((layer, index) => {
    if (layer.type === "fill") finalFillIndex = index;
  });
  return finalFillIndex >= 0 ? layers[finalFillIndex + 1]?.id : undefined;
}
