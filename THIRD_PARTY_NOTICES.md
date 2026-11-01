# Third-party notices

Original project contributions, research data, and imported software have
different licenses. See [LICENSING.md](LICENSING.md) for scope. Preserve all
upstream copyright notices and license texts when redistributing this source.

## Directly inspected source components

| Component | Existing notice or license | Included text or source |
|---|---|---|
| Upstream libuipc | Apache License 2.0 | [`libuipc-diff/LICENSE`](libuipc-diff/LICENSE) |
| Upstream Python package directory | GNU General Public License v3 text | [`libuipc-diff/python/LICENSE`](libuipc-diff/python/LICENSE) |
| MUDA | Apache License 2.0 | [`libuipc-diff/external/muda/LICENSE`](libuipc-diff/external/muda/LICENSE) |
| attcs Octree | MIT; copyright 2021 attcs | [`libuipc-diff/external/octree/Octree/LICENSE`](libuipc-diff/external/octree/Octree/LICENSE) |
| TetGen | AGPL version 3 or later, or a separately obtained commercial license | [`libuipc-diff/external/tetgen/LICENSE`](libuipc-diff/external/tetgen/LICENSE) |
| Included NVTX headers | NVIDIA-specific notices; not covered by this project's AGPL grant | [`libuipc-diff/external/nvtx3/nvtx3/nvToolsExt.h`](libuipc-diff/external/nvtx3/nvtx3/nvToolsExt.h) and adjacent headers |

The project does not supply a commercial TetGen license. Its current CMake
configuration links TetGen into the geometry library. Keep its AGPL terms in
mind when distributing the combined backend; see the
[TetGen licensing FAQ](https://www.wias-berlin.de/software/tetgen/FAQ-license.jsp).

## NVIDIA NVTX notice

The older NVTX headers included in this source snapshot identify their source
as proprietary NVIDIA material and refer to NVIDIA software license terms.
Do not interpret the root AGPL license as permission to relicense those headers.
The applicable redistribution permission and compatibility with a combined
AGPL program remain to be verified before publishing that combined program.

The following notice is retained from the bundled `nvToolsExt.h` for documentation:

```text
Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.

NOTICE TO USER:

This source code is subject to NVIDIA ownership rights under U.S. and
international Copyright laws.

This software and the information contained herein is PROPRIETARY and
CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
of a form of NVIDIA software license agreement.

NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
OR PERFORMANCE OF THIS SOURCE CODE.

U.S. Government End Users.   This source code is a "commercial item" as
that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
"commercial computer  software"  and "commercial computer software
documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
and is provided to the U.S. Government only as a commercial end item.
Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
source code with only those rights set forth herein.

Any use of this source code in individual and commercial software must
include, in the user documentation and internal comments to the code,
the above Disclaimer and U.S. Government End Users Notice.
```

## Build-time and runtime dependencies

Dependencies obtained through vcpkg, Python packages, the CUDA toolkit, and the
NVIDIA driver retain their respective licenses. The table above is an inventory
of the directly inspected bundled components, not a complete inventory of every
transitive dependency. When distributing binaries, retain required notices and
license files for all redistributed dependencies, including applicable files
under `vcpkg_installed/<triplet>/share/<port>/copyright`.

No existing upstream copyright notice is replaced by the root license files.
