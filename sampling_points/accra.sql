WITH line        AS  (SELECT osmid,
                             highway,
                             lanes,
                             junction,
                             maxspeed,
                             name,
                             oneway,
                             ref,
                             service,
                             tunnel,
                             width,
                             (ST_Dump(the_geom_webmercator)).geom AS geom
                        FROM accra_roads),
     linemeasure AS  (SELECT ST_AddMeasure(line.geom, 0, ST_Length(line.geom)) AS linem,
                             generate_series(0, ST_Length(line.geom)::int, 100) AS i,
                             osmid,
                             highway,
                             lanes,
                             junction,
                             maxspeed,
                             name,
                             oneway,
                             ref,
                             service,
                             tunnel,
                             width
                        FROM line),
      geometries AS ( SELECT i,
                             osmid,
                             highway,
                             lanes,
                             junction,
                             maxspeed,
                             name,
                             oneway,
                             ref,
                             service,
                             tunnel,
                             width,
                             (ST_Dump(ST_GeometryN(ST_LocateAlong(linem, i), 1))).geom AS geom 
                        FROM linemeasure),

          points AS  (SELECT ROW_NUMBER() OVER (ORDER BY 1) AS cartodb_id,
                             ST_SRID(geom) AS srid,
                             ST_Transform(ST_SetSRID(ST_MakePoint(ST_X(geom), ST_Y(geom)), ST_SRID(geom)), 4326) AS the_geom,
                             ST_Transform(ST_SetSRID(ST_MakePoint(ST_X(geom), ST_Y(geom)), ST_SRID(geom)), 3857) AS the_geom_webmercator,
                             osmid,
                             highway,
                             lanes,
                             junction,
                             maxspeed,
                             name,
                             oneway,
                             ref,
                             service,
                             tunnel,
                             width
                        FROM geometries)
             SELECT cartodb_id,
                    srid,
                    the_geom,
                    the_geom_webmercator,
                    ST_X(the_geom) AS lat,
                    ST_Y(the_geom) AS lng,
                    osmid,
                    highway,
                    lanes,
                    junction,
                    maxspeed,
                    name,
                    oneway,
                    ref,
                    service,
                    tunnel,
                    width
               FROM points