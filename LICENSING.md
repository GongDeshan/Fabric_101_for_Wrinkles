# Licensing

This repository contains separately licensed software, research data, and
third-party source code. The licenses below apply only to rights that the
respective contributors are entitled to grant. Existing third-party copyright
and license notices remain in effect.

## Original software contributions

Unless a file carries another applicable notice, original software contributions
made by this project's contributors are licensed under the **GNU Affero General
Public License, version 3 only** (`AGPL-3.0-only`). The full text is in [LICENSE](LICENSE).

This grant covers project-authored processing and training scripts, build
support, and original modifications to the simulation backend. Imported upstream
code retains its existing license; this statement does not relicense it.
Copyright remains with the respective contributors.

AGPL-3.0 is not a noncommercial license. Commercial use of covered software is
permitted subject to its terms, including the applicable source-distribution
obligations. Modified versions supporting remote user interaction must comply
with section 13's source-offer requirement. The data license below does not add
a noncommercial restriction to the software.

## Data, photographs, and catalog

Except for material identified as belonging to a third party or carrying a
different license, the project-provided research data and photographs are
licensed under **Creative Commons Attribution-NonCommercial 4.0 International**
(`CC-BY-NC-4.0`). The full text is in [LICENSE-DATA](LICENSE-DATA), with the
[official explanation and legal code](https://creativecommons.org/licenses/by-nc/4.0/).

This applies to:

- Raw measurements in `data/`.
- Microscopy photographs in `photos/`.
- Processed measurement curves in `processed_data/` and data plots in `plots/`.
- `fabric_catalog.csv`, `data_folder_mapping.csv`, and the catalog in `FABRICS.md`.

For rights covered by this license, sharing and adaptation are permitted for
noncommercial purposes with appropriate attribution, a license reference, and
an indication of changes. Commercial use requires separate permission from the
relevant rights holders. The meaning of NonCommercial is defined in section 1
of the license; it is not determined solely by whether the user is a company or
a university.

For attribution, identify **Fabric-101 / Learnable Persistent Wrinkle Formation
in Cloth Simulation**, retain any supplied creator credits, provide a link to
the repository where reasonably practicable, and identify the CC BY-NC 4.0
license and any modifications. For scholarly work, please also cite the paper
listed in the README. The citation request adds no restriction beyond the
applicable license.

These licenses do not grant rights in separately published papers, third-party
illustrations, logos, or other material that this project does not own. They do
not restrict uses for which permission is not required by applicable law.

## Upstream and third-party components

Keep the existing license files and copyright notices in `libuipc-diff/` and its
dependencies. In particular, the Apache-2.0 license accompanying upstream
libuipc, the GPLv3 notice in its Python directory, the TetGen licensing terms,
and other dependency notices remain applicable. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the directly inspected components.

The root AGPL license is not a claim that every bundled file has been relicensed
under AGPL. Redistribution of a combined program must satisfy all applicable
licenses. In particular, the included older NVTX headers carry NVIDIA-specific
terms that require separate verification before redistributing a combined
backend containing them. Their inclusion here is not a claim of AGPL
compatibility or an additional redistribution grant.

The applicable licenses contain their own warranty disclaimers. This document
explains scope and does not replace or modify their legal terms.
