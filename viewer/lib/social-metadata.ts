import type { Metadata } from "next";

// The public address is www.todd.sh/mapscan (todd.sh proxies this deployment).
// Next does not prepend basePath to metadata URLs, so paths passed here are
// the full public paths, e.g. "/mapscan/map".
export const siteOrigin = new URL("https://www.todd.sh");

const previewImage = {
  url: "https://www.todd.sh/project-previews/mapscan-palette-v2.webp",
  width: 1586,
  height: 992,
  alt: "Layered paper maps of California converging into alignment inside a registration frame",
};

export function socialMetadata({
  path,
  title,
  description,
}: {
  path: string;
  title: string;
  description: string;
}): Pick<Metadata, "alternates" | "openGraph" | "twitter"> {
  return {
    alternates: { canonical: path },
    openGraph: {
      type: "article",
      siteName: "Todd Sherman",
      title,
      description,
      url: path,
      images: [previewImage],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [previewImage],
      creator: "@tdd",
    },
  };
}
