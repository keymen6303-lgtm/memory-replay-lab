"""One-seed smoke example. Does not estimate confirmation-study effects."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from src.energy import fixture, build_model, Memory
from src.fidelity import source_vectors, construct, recognize

def main():
    with threadpool_limits(limits=1):
        f = fixture(101, 24, .35)
        sources = source_vectors(f, 101)
        base = build_model(f, 'no_replay')[0]
        rows = []
        for source in ('exact', 'coherent', 'independent'):
            model, ids, events = construct(f, source, 'high_energy', sources)
            before = model.fingerprint()
            scores = recognize(model, f['query'])
            recovered, steps, residual, converged = model.recall(f['cues'])
            parents = np.array([r['parent'] for r in f['cue_info']])
            identity = ((recovered @ f['x'].T).argmax(axis=1) == parents) & converged
            assert before == model.fingerprint()
            assert len(events) == 64 and len(ids) == 8
            row = dict(source=source, stored_vectors=len(model.vectors),
                unique_support=len(np.unique(model.vectors,axis=0)),
                array_bytes=model.vectors.nbytes+model.masses.nbytes,
                partial_identity_fraction=float(identity.mean()),
                all_partial_converged=bool(converged.all()))
            if source == 'coherent':
                weights = np.ones(25); weights[ids] += 2
                collapsed = Memory(f['y'], weights)
                error = float(np.max(np.abs(scores[0]+collapsed.energy(f['query']))))
                assert error < 1e-12
                max_error = float(np.max(np.abs(scores[1]-recognize(base,f['query'])[1])))
                assert max_error < 1e-12
                row.update(weighted_collapse_max_error=error, max_similarity_base_error=max_error)
            rows.append(row)
        print(json.dumps(dict(seed=101, scope='one-seed demonstration, not confirmation evidence', results=rows),indent=2))

if __name__ == '__main__':
    main()
