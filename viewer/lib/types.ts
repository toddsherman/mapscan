export type CategoryManifest = {
  id: string;
  class_id: number;
  label: string;
  display_rgb: [number, number, number];
  pixel_count: number;
  tile_template?: string;
  tile_file_count?: number;
  render_mode?: "recolorable_mask" | "native_color";
  color_editable?: boolean;
  category_role?: "continuous_surface" | "continuous_band";
  default_enabled?: boolean;
  units?: string;
  value_range?: {
    lower_bound: number | null;
    upper_bound: number | null;
    lower_inclusive: boolean;
    upper_inclusive: boolean;
    special_value_ids: string[];
  };
  legend_stops?: Array<{
    value: number;
    label: string;
    display_rgb: [number, number, number];
  }>;
  special_values?: Array<{
    id: string;
    label: string;
    value: number;
    numeric_value_known: boolean;
    display_rgb: [number, number, number];
  }>;
};

export type IndexedRasterManifest = {
  encoding: "png_luma_alpha_uint8_class_id_v1";
  class_id_channel: "red_after_browser_png_decode";
  coverage_channel: "alpha";
  nodata_class_id: 0;
  class_id_range: [number, number];
  raster_color_mix: [number, number, number, number];
  raster_color_range: [number, number];
  tile_template: string;
  tile_file_count: number;
  tile_image_byte_count?: number;
  mean_tile_image_byte_count?: number;
  tile_set_sha256: string;
};

export type DatasetManifest = {
  status?: string;
  asset_base?: string;
  id: string;
  title: string;
  bounds: [number, number, number, number];
  center: [number, number];
  minimum_zoom: number;
  maximum_native_zoom: number;
  categorical_tile_encoding?: "per_class_rgba" | "indexed_class_id";
  overview?: {
    mode: string;
    supersampling: number;
    overview_zooms: [number, number];
    exact_binary_zoom: number;
  };
  source_image: string | null;
  continuous?: {
    units: string;
    encoding: Record<string, unknown>;
    ramp_stops: CategoryManifest["legend_stops"];
    special_values: CategoryManifest["special_values"];
    selection_bands?: Array<{
      id: string;
      label: string;
      lower_bound: number | null;
      upper_bound: number | null;
      lower_inclusive: boolean;
      upper_inclusive: boolean;
      display_rgb: [number, number, number];
      special_value_ids: string[];
      pixel_count: number;
    }>;
    value_raster?: {
      path: string;
      sha256: string;
      width: number;
      height: number;
    };
  };
  approval?: {
    status?: string;
    reviewed_at?: string | null;
    sha256?: string;
  };
  boundary?: {
    kind?: string;
    authority?: string;
    diagnostic_only?: boolean;
    continuous_border_component_count?: number;
    expected_boundary_component_count?: number;
    mainland_interior_pixel_count?: number;
    publication_interior_pixel_count?: number;
    colored_pixel_count_outside_boundary?: number;
    unclassified_pixel_count_inside_boundary?: number;
    geojson?: string;
    geojson_sha256?: string;
    geojson_vertex_count?: number;
    geojson_feature_count?: number;
    canonical_boundary_id?: string;
    raster?: string;
    raster_sha256?: string;
    raster_width?: number;
    raster_height?: number;
    raster_bounds?: [number, number, number, number];
  } | null;
  provenance?: {
    manifest: string;
    sha256: string;
  };
  layers: Array<{
    id: string;
    label: string;
    kind?: "categorical" | "continuous";
    categories: CategoryManifest[];
    indexed_raster?: IndexedRasterManifest;
  }>;
};

export type CategoryStyle = {
  enabled: boolean;
  color: string;
  opacity: number;
};

export type ViewerDataset = DatasetManifest & {
  public_path: string;
  menu_title: string;
};
