import type { ExpressionSpecification } from "mapbox-gl";
import type { CategoryManifest, CategoryStyle } from "@/lib/types";

const TRANSPARENT = "rgba(0, 0, 0, 0)";

function rgba(color: string, opacity: number) {
  const match = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color);
  if (!match) throw new Error(`Indexed raster color must be #RRGGBB: ${color}`);
  const [, red, green, blue] = match;
  const alpha = Math.min(1, Math.max(0, opacity));
  return `rgba(${Number.parseInt(red, 16)}, ${Number.parseInt(green, 16)}, ${Number.parseInt(blue, 16)}, ${alpha})`;
}

export function indexedRasterSourceId(layerId: string) {
  return `mapscan-indexed-source-${layerId}`;
}

export function indexedRasterLayerId(layerId: string) {
  return `mapscan-indexed-layer-${layerId}`;
}

export function indexedRasterCategoryLayerId(layerId: string, categoryId: string) {
  return `${indexedRasterLayerId(layerId)}-${categoryId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

export function indexedRasterCategoryColor(
  category: CategoryManifest,
  color: string,
): ExpressionSpecification {
  if (
    !Number.isInteger(category.class_id) ||
    category.class_id < 1 ||
    category.class_id > 255
  ) {
    throw new Error(
      `Indexed raster class id must be an integer from 1 to 255: ${category.class_id}`,
    );
  }
  return [
    "step",
    ["raster-value"],
    TRANSPARENT,
    (category.class_id - 0.5) / 255,
    rgba(color, 1),
    (category.class_id + 0.5) / 255,
    TRANSPARENT,
  ];
}

/**
 * Build a Mapbox raster-color ramp for lossless uint8 class IDs.
 *
 * The browser expands PNG luma-alpha to RGBA. With raster-color-mix=[1,0,0,0]
 * and raster-color-range=[0,1], an encoded class id N is read as N/255.
 * Half-integer steps isolate every ID while the returned RGBA alpha implements
 * each class's independent toggle and opacity. Source alpha still preserves
 * NoData and fractional low-zoom coastline coverage.
 */
export function indexedRasterColor(
  categories: CategoryManifest[],
  styles: Record<string, CategoryStyle>,
): ExpressionSpecification {
  const byClassId = new Map<number, CategoryManifest>();
  for (const category of categories) {
    if (!Number.isInteger(category.class_id) || category.class_id < 1 || category.class_id > 255) {
      throw new Error(`Indexed raster class id must be an integer from 1 to 255: ${category.class_id}`);
    }
    if (byClassId.has(category.class_id)) {
      throw new Error(`Indexed raster class id is duplicated: ${category.class_id}`);
    }
    byClassId.set(category.class_id, category);
  }
  const maximumClassId = Math.max(0, ...byClassId.keys());
  const expression: unknown[] = ["step", ["raster-value"], TRANSPARENT];
  for (let classId = 1; classId <= maximumClassId; classId += 1) {
    const category = byClassId.get(classId);
    const style = category ? styles[category.id] : undefined;
    const output =
      category && style?.enabled && style.opacity > 0
        ? rgba(style.color, style.opacity)
        : TRANSPARENT;
    expression.push((classId - 0.5) / 255, output);
  }
  expression.push((maximumClassId + 0.5) / 255, TRANSPARENT);
  return expression as ExpressionSpecification;
}
