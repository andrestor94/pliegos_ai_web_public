# main.py (ejemplo)
from fastapi import APIRouter, HTTPException
from database import eliminar_del_historial_validado, valorar_y_eliminar_por_id

router = APIRouter()

@router.delete("/api/analisis/{timestamp}")
def delete_analisis(timestamp: str):
    ok, code = eliminar_del_historial_validado(timestamp)
    if ok:
        return {"ok": True}
    if code == "REQUIRES_RATING":
        # 409 para que el front abra el modal de valoración
        raise HTTPException(status_code=409, detail={"requires_rating": True})
    if code == "NOT_FOUND":
        # idempotente: 204/200 también sería válido; aquí 404 explícito
        raise HTTPException(status_code=404, detail="No encontrado")
    raise HTTPException(status_code=500, detail="Error al eliminar")

@router.post("/api/analisis/{hid}/rate-and-delete")
def rate_and_delete(hid: int, payload: dict):
    score = payload.get("score")
    ok, code = valorar_y_eliminar_por_id(hid, score)
    if ok:
        return {"ok": True}
    if code == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="No encontrado")
    raise HTTPException(status_code=400, detail="No se pudo valorar y eliminar")
