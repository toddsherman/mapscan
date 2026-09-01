# MapScan share URLs

MapScan configurations are serialized into the `/mapscan/map` query string.
The address bar updates automatically after every composition change and after
the map finishes moving. **Copy link** copies that canonical URL. Opening it in
a fresh browser restores the composition and camera without a server-side saved
state.

## Version 3 parameters

- `config=3` identifies the current format.
- `state` is URI-safe LZ-compressed JSON. It contains the dataset open in the
  editor, every enabled or restyled layer, per-layer colors and opacity,
  non-default dataset opacity, the complete dataset order, and longitude,
  latitude, zoom, bearing, and pitch.
- Transient interface state such as the copied confirmation, an open source
  image dialog, and the alignment diagnostic is intentionally excluded because
  it is not part of the rendered map composition.

Unknown dataset or layer IDs and invalid values are ignored. Version 3 remains
compatible with version 2 and both earlier MapScan link formats, including
three-part multi-dataset layer entries and four-part single-dataset entries.

## Preparing article examples

1. Open `/mapscan/map` and clear the starting composition.
2. Select and style the layers needed for the example.
3. Set each dataset's opacity and use its arrows to establish the stack order.
4. Pan and zoom to frame the observation.
5. Copy the already-current browser URL or click **Copy link**, then save it
   with the example's title and explanation.

The article can link directly to each saved configuration without maintaining a
separate preset file.
