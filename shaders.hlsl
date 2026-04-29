struct VSOut { float4 pos : SV_Position; float3 col : COLOR; };

VSOut VSMain(uint id : SV_VertexID)
{
    float2 pos[3] = { float2(0, 0.5), float2(0.5, -0.5), float2(-0.5, -0.5) };
    float3 col[3] = { float3(1,0,0), float3(0,1,0), float3(0,0,1) };
    VSOut o;
    o.pos = float4(pos[id], 0, 1);
    o.col = col[id];
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    return float4(i.col, 1);
}