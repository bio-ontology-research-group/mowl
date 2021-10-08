package org.mowl.Onto2Vec;

import org.semanticweb.owlapi.util.ShortFormProvider;
import org.semanticweb.owlapi.model.OWLEntity;

public class Onto2VecShortFormProvider implements ShortFormProvider {

    @Override
    public String getShortForm(OWLEntity entity) {
        return entity.toString();
    }

    @Override
    public void dispose() {}
    
}
